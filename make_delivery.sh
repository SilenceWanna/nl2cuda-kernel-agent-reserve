#!/usr/bin/env bash
# make_delivery.sh —— 把"最小完整交付物"（skill 包）从本仓库汇集到一个独立目录，方便查看/分发。
#
# 交付物 = 一个能把"自然语言算法描述 → CUDA 前反向 .cu"的 skill 包。只含通用用户跑通全流程
# 真正需要的文件，排除作者专用测试基础设施、内部测试记录、私有笔记、编译缓存。
#
# 用法：
#   bash make_delivery.sh [目标目录] [--push] [--repo <git-url>] [--keep-docs]
#     目标目录默认 ../nl2cuda-delivery（仓库外，不污染本仓、无文件重复）。
#     --push        生成+净化+自检全部通过后，在目标目录全新 git init → 单条 commit →
#                   force-push 到交付仓（--repo 指定，默认下方 DEFAULT_PUSH_REPO）。
#                   交付仓是"单条干净历史"的分发仓（每次重生都是一条全新 commit，故 force-push），
#                   与作者的开发/测试仓(reserve)物理隔离——绝不共享 .git 历史，防私有内容经 git log 泄露。
#     --repo <url>  交付仓地址（仅 --push 时用）。
#     --keep-docs   README.md/USAGE.md 从交付仓远程取回（保留手维护的精简版），不用开发仓生成的覆盖。
#                   用于：docs 在交付仓侧手维护，但方法论/framework/cases 要从开发仓同步更新时。
#
# 产物是本仓文件的一份拷贝快照。不带 --push 时是普通目录（查看/打包用）；带 --push 时额外推到交付仓。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR"                                   # 本仓库根
DEFAULT_PUSH_REPO="git@coding.jd.com:ads-model-example/nl2cuda-kernel-skill.git"

# ---- 参数（位置参数=目标目录；--push/--repo/--keep-docs 为可选 flag）----
DEST=""
DO_PUSH=0
KEEP_DOCS=0
PUSH_REPO="$DEFAULT_PUSH_REPO"
while [ $# -gt 0 ]; do
  case "$1" in
    --push)      DO_PUSH=1; shift ;;
    --repo)      PUSH_REPO="$2"; shift 2 ;;
    --keep-docs) KEEP_DOCS=1; shift ;;   # README.md/USAGE.md 从交付仓远程取回(保留手维护版),不用开发仓生成的
    -*)          echo "未知参数: $1" >&2; exit 2 ;;
    *)           DEST="$1"; shift ;;
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

# ---- 改写仓地址：开发仓源文件里 clone URL/目录名/标题是旧仓名 nl2cuda-kernel-agent（开发仓自己叫这个名合理），
#      但交付物要指向交付仓 coding.jd.com 的 cuda-agent-skill。三段改写（顺序敏感）：
#        ① 完整 clone URL（github https，裸名或 -skill 变体）→ coding SSH 全地址；
#        ② 残留的 -skill 显式引用 → 交付目录名 cuda-agent-skill；
#        ③ 残留的裸名（cd 目录名/标题）→ cuda-agent-skill。
#      ②先于③：否则③的裸名正则会把 nl2cuda-kernel-agent-skill 前缀改成 cuda-agent-skill-skill。
DELIVERY_REPO_URL="git@coding.jd.com:ads-model-example/nl2cuda-kernel-skill.git"
DELIVERY_DIR_NAME="nl2cuda-kernel-skill"
find "$DEST" -type f \( -name '*.md' -o -name '*.ipynb' -o -name '*.py' -o -name '*.sh' -o -name '*.txt' \) -print0 \
  | while IFS= read -r -d '' f; do
      python - "$f" "$DELIVERY_REPO_URL" "$DELIVERY_DIR_NAME" <<'PYEOF'
import re, sys
path, url, dirname = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, encoding="utf-8") as fh:
    txt = fh.read()
orig = txt
# ① 完整 clone URL（裸名或 -skill）→ coding SSH 全地址
txt = re.sub(r"https://github\.com/SilenceWanna/nl2cuda-kernel-agent(?:-skill)?\.git", url, txt)
# ② 残留的 -skill 显式引用 → 交付目录名
txt = re.sub(r"nl2cuda-kernel-agent-skill", dirname, txt)
# ③ 残留的裸名（其后非连字符，避免误伤已处理的复合名）→ 交付目录名
txt = re.sub(r"nl2cuda-kernel-agent(?!-)", dirname, txt)
if txt != orig:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)
PYEOF
    done

# ---- 清 notebook 里引用交付仓不存在 case 的 cell（E2E #16）----
#      开发仓 run.ipynb 含 rbf + layernorm 两个 case 的演示 cell；交付仓只保留 rbf 样例，
#      layernorm cell 会指向不存在的 cases.layernorm → ModuleNotFoundError。删掉这些 cell（及其前导
#      markdown 说明），只留 rbf 冒烟。开发仓 notebook 不动（作者环境有 layernorm）。
if [ -f "$DEST/notebooks/run.ipynb" ]; then
  python - "$DEST/notebooks/run.ipynb" <<'PYEOF'
import json, sys, re
path = sys.argv[1]
nb = json.load(open(path, encoding="utf-8"))
kept = []
for c in nb.get("cells", []):
    src = "".join(c.get("source", []))
    # 删：引用非 rbf case 的 code cell（--case layernorm 等）+ 其"性能测试…LayerNorm case"前导 markdown
    is_nonrbf_code = c.get("cell_type") == "code" and re.search(r"--case\s+(?!rbf\b)[a-z_]+", src)
    is_layernorm_md = c.get("cell_type") == "markdown" and ("layernorm" in src.lower() or "LayerNorm" in src)
    if is_nonrbf_code or is_layernorm_md:
        continue
    # 去掉 rbf 冒烟注释里的样例实测加速比数字（交付物不出现验证结果数字）
    if c.get("cell_type") == "code":
        c["source"] = [re.sub(r"冒烟：加速比（rbf[^）]*）", "冒烟：跑通计时基准即可", ln)
                       for ln in c.get("source", [])]
    kept.append(c)
nb["cells"] = kept
json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"[notebook] 交付仓 run.ipynb 保留 {len(kept)} cell（删非 rbf case 引用）")
PYEOF
fi

# ---- 净化：把交付副本里方法论/约定文件的作者私有自测内容(run_on_a100/双跳SSH等)洗成通用 verify/bench ----
#      主仓正文保持原样(=作者开发/测试仓)，只改目标目录副本。规则失效(drift)或有残留会 exit 1。
if ! PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python "$SRC/skill/scripts/_sanitize_for_delivery.py" "$DEST"; then
  echo "[make_delivery] ✗ 净化失败(见上),交付物未通过一般性检查" >&2; exit 1
fi

# ---- --keep-docs：README.md/USAGE.md 从交付仓远程取回(保留手维护版)，覆盖开发仓生成的 ----
#      场景：README/USAGE 在交付仓侧手维护(手动精简)，方法论/framework/cases 仍从开发仓生成。
#      放在净化后、自检/泄露扫描前——取回的 docs 一并过下方兜底扫描，不漏私有词。
if [ "$KEEP_DOCS" = "1" ]; then
  echo "[make_delivery] === --keep-docs：从交付仓远程取回 README/USAGE ===" >&2
  TMP_DOCS="$(mktemp -d)"
  if git clone -q --depth 1 "$PUSH_REPO" "$TMP_DOCS" 2>/dev/null; then
    for d in README.md USAGE.md; do
      if [ -f "$TMP_DOCS/$d" ]; then cp "$TMP_DOCS/$d" "$DEST/$d"; echo "  ✓ 取回 $d" >&2
      else echo "  ⚠ 远程无 $d，保留生成版" >&2; fi
    done
  else
    echo "[make_delivery] ✗ --keep-docs 无法 clone 交付仓 $PUSH_REPO 取 docs" >&2; rm -rf "$TMP_DOCS"; exit 1
  fi
  rm -rf "$TMP_DOCS"
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
    # commit 作者用推送者本人的 git 全局身份（user.name/user.email）；须先 git config 配好，否则 commit 报错。
    git -c commit.gpgsign=false \
        commit -q -m "sync: 从开发仓净化生成交付版(方法论+framework+rbf样例;README/USAGE见仓库历史)"
    git remote add origin "$PUSH_REPO"
    git push -f -u origin main
  ) || { echo "[make_delivery] ✗ 推送失败" >&2; exit 1; }
  echo "[make_delivery] ✓ 已 force-push 到交付仓（单条干净历史）" >&2
fi
