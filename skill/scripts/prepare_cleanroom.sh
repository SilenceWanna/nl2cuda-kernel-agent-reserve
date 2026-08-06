#!/usr/bin/env bash
# prepare_cleanroom.sh —— 三宿主测试"干净房间"准备器（堵答案泄露：agent 只能靠精简 description + 通用方法论自写）。
#
# 背景（血泪教训，2026-07-30 gptme cholesky）：仅删目标 case 的实现文件**不够干净**——
# 仓库里 skill/AGENT_TEST_MATRIX.md 逐 case 记录了每个宿主的解法/公式/性能数字，
# README.md 的形态光谱表也含解法要点，其他 case 的实现可被抄结构，delivery/ 是完整交付代码。
# agent 一旦 grep 这些文件就等于"开卷考"，测出的是抄袭而非真实能力（gptme 实测 grep 命中矩阵里
# cholesky 的完整 Φ 算子公式 + codex 解法 + 0.31/1.25 性能，本轮作废）。
#
# 本脚本产出真·干净房间：只留【目标 case 的精简 description】+【通用方法论(SKILL/loop/DESIGN/
# CONVENTIONS/AGENTS 的算法无关技巧)】+【framework】。剥离所有"逐 case 具体答案"泄露源：
#   1. skill/AGENT_TEST_MATRIX.md（逐 case 解法/公式/性能——最严重泄露源）→ 删
#   2. 所有**非目标 case** 的实现（reference/config/__init__/op/kernels）→ 清空（防抄近邻结构）
#   3. 所有 cases/*/delivery/（完整独立编译交付代码）→ 删
#   4. README.md 的形态光谱表 + 核心结论（含解法要点/性能）→ 替换为中性占位
#   5. 目标 case 自身的实现 → 清空为 0 字节（只留 description.md）
# 然后建 cleanroom 分支并 commit 这个空状态（HEAD 里实现全 0 字节，agent 无论怎么 git 操作都还原不出）。
#
# ⚠️ 关于 SKILL.md：它是 agent 必读的**通用方法论**，保留。其"判据总纲/边界识别"含**方向性**元规律
#   （如"线性求解反向=解伴随""归一化前向可能带宽墙"）——这是 skill 应给的指引，非可直接抄的完整公式/kernel。
#   泄露的严重性在于矩阵有**完整公式 + 宿主实现细节 + 性能数字**；SKILL 只给方向，二者性质不同，故 SKILL 不净化。
#   （若某形态名在 SKILL 里带了过细的解法，可 --scrub-skill 追加净化，默认不动。）
#
# 用法：
#   bash skill/scripts/prepare_cleanroom.sh --case cholesky --dir /path/to/cleanroom [--desc <file>]
#     --case  目标 case 名（保留其 description，清空其实现）
#     --dir   干净房间输出目录（会 git clone 到此；已存在则先删）
#     --desc  可选：用该文件覆盖目标 case 的 description.md（写精简算法定义；不给则保留仓库原 description）
#     --repo  可选：clone 源（默认 GitHub 公共仓库）
#     --branch 可选：cleanroom 分支名（默认 cleanroom-<case>）
#
# 产出后，把 --dir 作为宿主 agent 的工作目录，喂精简算法定义即可（agent grep 不到任何 case 的答案）。
set -uo pipefail

CASE=""
DIR=""
DESC=""
REPO="https://github.com/SilenceWanna/nl2cuda-kernel-agent.git"
BRANCH=""

while [ $# -gt 0 ]; do
  case "$1" in
    --case)   CASE="$2"; shift 2 ;;
    --dir)    DIR="$2"; shift 2 ;;
    --desc)   DESC="$2"; shift 2 ;;
    --repo)   REPO="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

[ -z "$CASE" ] && { echo "缺 --case" >&2; exit 2; }
[ -z "$DIR" ]  && { echo "缺 --dir" >&2; exit 2; }
[ -z "$BRANCH" ] && BRANCH="cleanroom-$CASE"

# tar/git 用的 PATH（Windows Git Bash 下 System32 兜底；无害于 Linux/WSL）
export PATH="$PATH:/c/Windows/System32"

echo "[cleanroom] clone $REPO → $DIR" >&2
rm -rf "$DIR"
git clone -q --depth 1 "$REPO" "$DIR" || { echo "[cleanroom] clone 失败" >&2; exit 1; }
cd "$DIR" || exit 1

# ---- 封存 git 历史（否则 agent 可 git log -p / git show 从旧提交还原目标 case 的完整实现，等于开卷）----
#      弃掉 clone 带来的全部历史与 origin，重建为无历史的空仓；后面在此之上 commit 净化后的空状态，
#      使 HEAD 与历史都不含答案，agent 无论怎么 git 操作都还原不出。
rm -rf .git
git init -q

# 目标 case 必须存在
[ -d "cases/$CASE" ] || { echo "[cleanroom] cases/$CASE 不存在" >&2; exit 1; }

# ---- 1. 删逐 case 解法/测试结果泄露源 ----
#      AGENT_TEST_MATRIX.md 逐 case 记录宿主解法/公式/性能；CASE_EVIDENCE.md 是 SKILL 的逐 case 实测例证附录；
#      MULTIAGENT_TEST_RESULTS.md 记录各宿主对某 case 的测试结果/性能。三者都含"某 case 用什么手段达到多少
#      加速比"，agent 读到=开卷考。SKILL 正文只留通用原理，保留。
rm -f skill/AGENT_TEST_MATRIX.md skill/CASE_EVIDENCE.md skill/MULTIAGENT_TEST_RESULTS.md
# 工作计划.md（进度日志逐 case 记录解法/性能/优化手段）+ 工作目标.md（私有）——同属逐 case 答案泄露源，删。
rm -f 工作计划.md 工作目标.md

# ---- 2+5. 精简 cases/：只留【目标 case】(留 description、清空实现) + 【rbf】(完整,作结构模板) ----
#      其余所有非目标、非 rbf 的 case 目录**整删**——它们的 description 可能含解法公式（如 cholesky 的
#      Φ 算子/伴随公式）、其实现可被抄近邻结构，对目标 case 测试是纯泄露面，删干净最省心。
#      rbf 保留完整：它是 SKILL/CLAUDE 明示的"照 cases/rbf/__init__.py 写 Case(...)"结构模板，且 rbf
#      （成对距离）与任何目标算法都不同，看它的实现无法抄出目标解法，不构成答案泄露。
#      目标 case：清空 reference/config/__init__/op/kernels（预建 0 字节解 file-not-found 卡顿），只留 description.md。
for cdir in cases/*/; do
  c="$(basename "$cdir")"
  if [ "$c" = "$CASE" ]; then
    for f in reference.py config.py __init__.py op.py; do
      [ -e "$cdir$f" ] && : > "$cdir$f"
    done
    if [ -d "${cdir}kernels" ]; then
      for cu in "${cdir}"kernels/*.cu; do
        [ -e "$cu" ] && : > "$cu"
      done
    fi
  elif [ "$c" = "rbf" ]; then
    :   # 保留完整 rbf 作结构模板
  else
    rm -rf "$cdir"
  fi
done

# ---- 3. 删所有 delivery/（完整独立编译交付代码）----
rm -rf cases/*/delivery

# ---- 4. README.md 换中性占位（原含形态光谱表 + 解法要点 + 性能数字）----
cat > README.md <<'EOF'
# nl2cuda-kernel-agent

自然语言算法描述 → CUDA 前向+反向 kernel 的 skill。以 PyTorch 参考实现为金标准过正确性验证，
规范计时下超过 `torch.compile`。算法无关，每个算法是 `cases/<name>/` 下一个 case。

> 本目录是**三宿主测试干净房间**：已剥离所有 case 的实现与历史测试记录，仅保留通用方法论
> （`skill/SKILL.md` 等）与目标 case 的算法描述。请按 `CONVENTIONS.md`/`AGENTS.md`/`CLAUDE.md`
> 与 `skill/SKILL.md` 的协议，从零实现目标 case。
EOF

# ---- 可选：用精简 description 覆盖目标 case ----
if [ -n "$DESC" ]; then
  [ -f "$DESC" ] || { echo "[cleanroom] --desc 文件不存在: $DESC" >&2; exit 1; }
  cp "$DESC" "cases/$CASE/description.md"
  echo "[cleanroom] 已用 $DESC 覆盖 cases/$CASE/description.md" >&2
fi

# ---- 建 cleanroom 分支并提交空状态（HEAD 里实现全 0 字节）----
git checkout -q -b "$BRANCH"
git add -A
git commit -q -m "cleanroom($CASE): 剥离矩阵/README/其他case实现/delivery，仅留精简description+通用方法论" \
  || { echo "[cleanroom] 无改动可提交（异常）" >&2; exit 1; }

# ---- 自检：确认泄露源已清 ----
echo "[cleanroom] === 自检 ===" >&2
FAIL=0
[ -f skill/AGENT_TEST_MATRIX.md ] && { echo "  ✗ AGENT_TEST_MATRIX.md 仍在" >&2; FAIL=1; }
[ -f skill/CASE_EVIDENCE.md ] && { echo "  ✗ CASE_EVIDENCE.md 仍在" >&2; FAIL=1; }
[ -f skill/MULTIAGENT_TEST_RESULTS.md ] && { echo "  ✗ MULTIAGENT_TEST_RESULTS.md 仍在" >&2; FAIL=1; }
[ -f 工作计划.md ] && { echo "  ✗ 工作计划.md 仍在（进度日志逐 case 记录解法/性能）" >&2; FAIL=1; }
[ -f 工作目标.md ] && { echo "  ✗ 工作目标.md 仍在" >&2; FAIL=1; }
[ -s "cases/$CASE/reference.py" ] && { echo "  ✗ 目标 case reference 非空" >&2; FAIL=1; }
# 泄露判定用**结构性检查**而非数字正则：达标线 "1.05×"、rbf 参考值 "1.10×" 等是**合规方法论**，
# 必须保留，与某 case 的答案性能数字无法靠"含小数"区分（旧数字正则误杀 14 个方法论文件）。
# 逐 case 答案本就集中在上面已删的文件（矩阵/证据附录/工作计划/工作目标）+ 各 case 实现里，
# 故只要"答案文件已删 + 所有实现清空 + delivery 删除 + 无非空 kernel"即干净。
# 下面再加一道**窄**的已知解法公式扫描（仅高辨识度、不会出现在方法论正文的完整公式片段）作纵深防御：
LEAK="$(git grep -liE 'grad_A=|L⁻ᵀ|Phi\(L|解伴随系统 Aᵀ' -- . ':!skill/SKILL.md' ':!skill/scripts/prepare_cleanroom.sh' 2>/dev/null || true)"
[ -n "$LEAK" ] && { echo "  ✗ 仍有文件含完整解法/性能泄露: $LEAK" >&2; FAIL=1; }
# 非空 kernel 检查：目标 case 已清空；rbf 作结构模板**有意保留完整**（不同算法，非答案），故豁免 rbf。
NONEMPTY_IMPL="$(find cases -name '*.cu' -not -path '*/delivery/*' -not -path 'cases/rbf/*' -size +0c 2>/dev/null | head -3)"
[ -n "$NONEMPTY_IMPL" ] && { echo "  ✗ 仍有非空 kernel: $NONEMPTY_IMPL" >&2; FAIL=1; }
if [ "$FAIL" = 0 ]; then
  echo "  ✓ 泄露源已清：矩阵+证据附录删除、所有 case 实现清空、delivery 删除、README 中性化" >&2
  echo "  ✓ 目标 case=$CASE 仅留 description（$(wc -c < cases/$CASE/description.md) 字节）" >&2
  echo "  ✓ SKILL.md 通用方法论保留（正文无点名 case 的完整解法/性能）" >&2
  echo "[cleanroom] 就绪：workdir=$DIR 分支=$BRANCH" >&2
else
  echo "[cleanroom] ✗ 自检未通过，请核查" >&2; exit 1
fi
