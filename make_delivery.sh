#!/usr/bin/env bash
# make_delivery.sh —— 把"最小完整交付物"（skill 包）从本仓库汇集到一个独立目录，方便查看/分发。
#
# 交付物 = 一个能把"自然语言算法描述 → CUDA 前反向 .cu"的 skill 包。只含通用用户跑通全流程
# 真正需要的文件，排除作者专用测试基础设施、内部测试记录、私有笔记、编译缓存。
#
# 用法：
#   bash make_delivery.sh [目标目录]
#     目标目录默认 ../nl2cuda-delivery（仓库外，不污染本仓、无文件重复）。
#
# 产物是本仓文件的一份拷贝快照，非 git 仓；查看/打包/分发用。清单见下方 INCLUDE/EXCLUDE。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR"                                   # 本仓库根
DEST="${1:-$SCRIPT_DIR/../nl2cuda-delivery}"        # 默认生成到仓库外同级目录

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
