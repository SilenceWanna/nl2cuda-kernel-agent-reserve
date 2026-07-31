#!/usr/bin/env bash
# make_delivery.sh —— 把"最小完整交付物"（skill 包）从本仓库汇集到一个独立目录，方便查看/分发。
#
# 交付物 = 一个能把"自然语言算法描述 → CUDA 前反向 .cu"的 skill 包。只含通用用户跑通全流程
# 真正需要的文件，排除作者专用测试基础设施、内部测试记录、私有笔记、编译缓存。
#
# 用法：
#   bash make_delivery.sh [目标目录] [--push] [--repo <git-url>]
#     目标目录默认 ../nl2cuda-delivery（仓库外，不污染本仓、无文件重复）。
#     --push        生成+净化+自检全部通过后，在目标目录全新 git init → 单条 commit →
#                   force-push 到交付仓（--repo 指定，默认下方 DEFAULT_PUSH_REPO）。
#                   交付仓是"单条干净历史"的分发仓（每次重生都是一条全新 commit，故 force-push），
#                   与作者的开发/测试仓(reserve)物理隔离——绝不共享 .git 历史，防私有内容经 git log 泄露。
#     --repo <url>  交付仓地址（仅 --push 时用）。
#
# 产物是本仓文件的一份拷贝快照。不带 --push 时是普通目录（查看/打包用）；带 --push 时额外推到交付仓。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR"                                   # 本仓库根
DEFAULT_PUSH_REPO="https://github.com/SilenceWanna/nl2cuda-kernel-agent-skill.git"

# ---- 参数（位置参数=目标目录；--push/--repo 为可选 flag）----
DEST=""
DO_PUSH=0
PUSH_REPO="$DEFAULT_PUSH_REPO"
while [ $# -gt 0 ]; do
  case "$1" in
    --push)  DO_PUSH=1; shift ;;
    --repo)  PUSH_REPO="$2"; shift 2 ;;
    -*)      echo "未知参数: $1" >&2; exit 2 ;;
    *)       DEST="$1"; shift ;;
  esac
done
[ -z "$DEST" ] && DEST="$SCRIPT_DIR/../nl2cuda-delivery"

# ---- 必含清单（通用用户从零到 .cu 真正需要的）----
#  framework/       算法无关评测基座（只读）
#  skill/*.md       方法论（SKILL/DESIGN/loop/AUTONOMOUS_LOOP；不含内部测试记录）
#  skill/scripts/   仅 4 个通用 CLI（不含作者专用 run_on_a100/autotest/start_gptme/prepare_cleanroom）
#  cases/rbf/       唯一保留的完整样例 case（agent 复制它作模板；含 delivery 独立编译成品样例）
#  约定文件/依赖/入口/说明书
INCLUDE_FILES=(
  "skill/SKILL.md" "skill/DESIGN.md" "skill/loop.md"
  "skill/USING_WITH_OTHER_AGENTS.md"
  "skill/scripts/verify_case.py" "skill/scripts/bench_case.py"
  "skill/scripts/profile_case.py" "skill/scripts/check_reference.py"
  "AGENTS.md" "CLAUDE.md" "CONVENTIONS.md"
  "requirements.txt" "scripts/probe_env.py" "notebooks/run.ipynb"
  "USAGE.md" "README.md"
)
INCLUDE_DIRS=(
  "framework"
  "cases/rbf"
)

# ---- 复制（去 __pycache__/.pyc）----
copy_one() {  # $1=相对路径（文件或目录）
  local rel="$1" s="$SRC/$1" d="$DEST/$1"
  if [ -d "$s" ]; then
    mkdir -p "$d"
    # 用 tar 管道保目录结构 + 排除缓存（find+cp 对空格路径易错）
    ( cd "$s" && find . -type f \
        -not -path '*/__pycache__/*' -not -name '*.pyc' -not -name '.gitkeep' -print0 ) \
      | while IFS= read -r -d '' f; do
          mkdir -p "$d/$(dirname "$f")"; cp "$s/$f" "$d/$f"
        done
  elif [ -f "$s" ]; then
    mkdir -p "$(dirname "$d")"; cp "$s" "$d"
  else
    echo "  ✗ 缺失(跳过): $rel" >&2; return 1
  fi
}

echo "[make_delivery] 源=$SRC" >&2
echo "[make_delivery] 目标=$DEST" >&2
rm -rf "$DEST"; mkdir -p "$DEST"
MISS=0
for p in "${INCLUDE_DIRS[@]}" "${INCLUDE_FILES[@]}"; do copy_one "$p" || MISS=1; done

# ---- 改写仓名：开发仓源文件里 clone URL/目录名/标题是旧仓名 nl2cuda-kernel-agent（开发仓自己叫这个名合理），
#      但交付物要指向交付仓 nl2cuda-kernel-agent-skill，否则用户照 USAGE clone 会（经 GitHub 改名重定向）
#      拿到开发/自测仓 reserve（含私有内容）。故在交付副本上全局改写（E2E #7）。
#      负向断言 (?!-) 保证不误伤已带后缀的名字（-skill/-reserve），幂等可重跑。
DELIVERY_REPO_SLUG="nl2cuda-kernel-agent-skill"
find "$DEST" -type f \( -name '*.md' -o -name '*.ipynb' -o -name '*.py' -o -name '*.sh' -o -name '*.txt' \) -print0 \
  | while IFS= read -r -d '' f; do
      python - "$f" "$DELIVERY_REPO_SLUG" <<'PYEOF'
import re, sys
path, slug = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as fh:
    txt = fh.read()
# 把裸旧名 nl2cuda-kernel-agent（其后不是 - 连字符，即未带 -skill/-reserve 后缀）改成交付仓名
new = re.sub(r"nl2cuda-kernel-agent(?!-)", slug, txt)
if new != txt:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new)
PYEOF
    done

# ---- 净化：把交付副本里方法论/约定文件的作者私有自测内容(run_on_a100/双跳SSH等)洗成通用 verify/bench ----
#      主仓正文保持原样(=作者开发/测试仓)，只改目标目录副本。规则失效(drift)或有残留会 exit 1。
if ! PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python "$SRC/skill/scripts/_sanitize_for_delivery.py" "$DEST"; then
  echo "[make_delivery] ✗ 净化失败(见上),交付物未通过一般性检查" >&2; exit 1
fi

# ---- 自检：确认排除项没混入 ----
echo "[make_delivery] === 自检 ===" >&2
FAIL=0
for bad in \
  "skill/scripts/run_on_a100.sh" "skill/scripts/autotest.sh" \
  "skill/scripts/start_gptme.sh" "skill/scripts/prepare_cleanroom.sh" \
  "skill/scripts/_sanitize_for_delivery.py" \
  "skill/AUTONOMOUS_LOOP.md" \
  "skill/AGENT_TEST_MATRIX.md" "skill/CASE_EVIDENCE.md" "skill/MULTIAGENT_TEST_RESULTS.md" \
  "工作计划.md" "工作目标.md"; do
  [ -e "$DEST/$bad" ] && { echo "  ✗ 排除项混入: $bad" >&2; FAIL=1; }
done
# 不应含 rbf 之外的 case
OTHER_CASES="$(find "$DEST/cases" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | grep -v '/rbf$' || true)"
[ -n "$OTHER_CASES" ] && { echo "  ✗ 含 rbf 之外的 case: $OTHER_CASES" >&2; FAIL=1; }
# 不应含编译缓存
PYC="$(find "$DEST" -name '__pycache__' -o -name '*.pyc' 2>/dev/null | head -1)"
[ -n "$PYC" ] && { echo "  ✗ 含编译缓存: $PYC" >&2; FAIL=1; }

if [ "$FAIL" = 0 ] && [ "$MISS" = 0 ]; then
  N="$(find "$DEST" -type f | wc -l | tr -d ' ')"
  echo "  ✓ 交付物就绪：$N 个文件，无排除项/无缓存/仅 rbf 样例 case" >&2
  echo "[make_delivery] 目录树：" >&2
  ( cd "$DEST" && find . -type f | sort | sed 's/^/    /' ) >&2
else
  echo "[make_delivery] ✗ 自检未通过（缺失=$MISS 混入=$FAIL），请核查" >&2; exit 1
fi

# ---- 通用 .gitignore（交付仓用；不含作者 a100/aider 私有条目）----
cat > "$DEST/.gitignore" <<'GI'
# Python
__pycache__/
*.py[cod]
*.egg-info/
# Virtual environments
venv/
.venv/
env/
.env
# PyTorch / CUDA build artifacts
*.so
*.o
*.obj
*.cubin
*.ptx
*.fatbin
torch_extensions/
# Test / cache / editor / OS
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.log
.vscode/
.DS_Store
GI

# ---- --push：全新 init → 单条 commit → force-push 到交付仓（干净单历史，与开发仓物理隔离）----
if [ "$DO_PUSH" = "1" ]; then
  echo "[make_delivery] === 推送到交付仓 $PUSH_REPO ===" >&2
  # 推送前最后一道兜底：全目录扫私有词（净化脚本已扫 .md，这里连非 .md 一起扫，双保险）
  LEAK="$(grep -rlE 'run_on_a100|AUTONOMOUS_LOOP|nl2cuda_gpu|11\.91\.|11\.127\.|start_gptme|prepare_cleanroom|AGENT_TEST_MATRIX|CASE_EVIDENCE|MULTIAGENT_TEST_RESULTS' "$DEST" 2>/dev/null || true)"
  if [ -n "$LEAK" ]; then
    echo "[make_delivery] ✗ 推送前扫描发现私有词残留，中止推送：" >&2
    echo "$LEAK" | sed 's#^#    #' >&2
    exit 1
  fi
  ( cd "$DEST" || exit 1
    rm -rf .git                                   # 弃任何旧 .git，保证全新历史（不带开发仓私有 log）
    git init -q
    git checkout -q -b main
    git add -A
    git -c user.name="nl2cuda-delivery" -c user.email="delivery@localhost" -c commit.gpgsign=false \
        commit -q -m "nl2cuda-kernel-agent (skill 交付版) — 从开发仓净化生成的通用交付版，使用见 USAGE.md"
    git remote add origin "$PUSH_REPO"
    git push -f -u origin main
  ) || { echo "[make_delivery] ✗ 推送失败" >&2; exit 1; }
  echo "[make_delivery] ✓ 已 force-push 到交付仓（单条干净历史）" >&2
fi
