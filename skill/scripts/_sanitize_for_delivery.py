#!/usr/bin/env python3
"""_sanitize_for_delivery.py —— 把交付物副本里方法论/约定文件的"作者私有自测内容"洗成通用表述。

背景（E2E_ISSUES #5）：主仓的 SKILL/loop/AGENTS/CLAUDE/CONVENTIONS 正文里嵌了作者专用的远程自测
基础设施引用（`run_on_a100.sh` 双跳 SSH 到 A100、`--sync-cli`/`--round-cap`、`AUTONOMOUS_LOOP.md`、
京东代理等）。这些对通用用户无意义、且会引导他们去跑一个不存在的私有脚本，破坏交付物一般性。

策略（用户定）：主仓保持原样（=作者的开发/测试仓），只在**交付物副本**上洗。本脚本由 make_delivery.sh
在文件已复制到目标目录后调用，就地改写目标目录里的这几个文件——不碰主仓。

设计要点：
- **失败即报**（drift 检测）：每条规则声明"必须命中的锚点"，若锚点在源文件里找不到（说明主仓正文改了、
  规则过期），脚本 exit 1 并指出哪条失效，绝不静默漏洗。
- **兜底扫描**：改写后再全量扫禁用词（run_on_a100/AUTONOMOUS_LOOP/--sync-cli/--round-cap/双跳/nl2cuda_gpu
  等），仍有残留就 exit 1。保证交付物里一个作者私有自测词都不剩。

用法：python skill/scripts/_sanitize_for_delivery.py <交付物根目录>
"""

import os
import re
import sys

# 通用自测表述（替换各约定文件"步骤4：自测"那一段的核心）——只用交付物里真实存在的通用 CLI。
GENERIC_SELFTEST = (
    "**自测（在你带 GPU 的环境；环境准备见 `USAGE.md`）**：先 "
    "`python skill/scripts/verify_case.py --case <name>`（正确性 allclose，须全 PASS；不过先修正确性、"
    "不看性能）→ 通过后 `python skill/scripts/bench_case.py --case <name>`（前反向各自 ≥1.05× "
    "`torch.compile` 才达标）。可选 `profile_case.py` 诊断瓶颈、`check_reference.py` 静态预检弱 baseline。"
)

# 每条规则：(文件相对路径, 锚点子串 anchor, 整行替换文本 replacement)
#   anchor 命中某行 → 该行整体替换为 replacement（replacement 为 None 表示删除该行）。
#   anchor 在文件里找不到 → drift，报错退出。
LINE_RULES = [
    # ---- AGENTS.md ----
    ("AGENTS.md",
     "即便你不建 bench.env，`run_on_a100.sh` 也会自动探测短核",
     "     **短核 case 务必让 config 规模支持 env 覆盖 + 建 `cases/<name>/bench.env` 声明放大规模**；自测时"
     "选一个计算主导区规模（baseline 前/反向 ≥1ms）避免短核假象（固定开销抬虚加速比）。"),
    ("AGENTS.md",
     "4. **自测（自动，无需用户提）**：跑 `bash skill/scripts/run_on_a100.sh",
     "4. " + GENERIC_SELFTEST),
    ("AGENTS.md",
     "   （首次加 `--sync-cli`）。它在远程 GPU 跑 verify+bench，末行给 `VERDICT=`。按 `skill/AUTONOMOUS_LOOP.md` 的",
     None),
    ("AGENTS.md",
     "   VERDICT 决策：`PASS`→交付；`VERIFY_FAIL`→修正确性（不看 bench）；`BENCH_FAIL`→按 `skill/loop.md` 优化未达标侧 kernel；",
     "   未达标（前或反向 <1.05×）按 `skill/loop.md` 优化未达标那侧的 kernel（先诊断瓶颈）；正确性未过先修正确性。"),
    ("AGENTS.md", "   `CV_INVALID`→原样重跑。", None),
    ("AGENTS.md",
     "**用户给算法定义 → 你读 SKILL.md → 自建 case → 写 reference/kernel/op（自主推导反向）→ run_on_a100.sh 自测 →",
     "**用户给算法定义 → 你读 SKILL.md → 自建 case → 写 reference/kernel/op（自主推导反向）→ 在 GPU 上 verify/bench 自测 →"),
    ("AGENTS.md",
     "## GPU 自测环境（已就绪，见 `skill/AUTONOMOUS_LOOP.md`）",
     "## GPU 自测环境"),
    ("AGENTS.md",
     "`run_on_a100.sh` 经双跳 SSH 直传远程 A100 跑评测。若你在 WSL，需先把密钥拷进 WSL：",
     "GPU 环境准备（Colab / 本地 GPU / 远程 SSH）见 `USAGE.md`。自测用 `python skill/scripts/verify_case.py --case <name>`"),
    ("AGENTS.md",
     "`cp /mnt/c/Users/<user>/.ssh/nl2cuda_gpu ~/.ssh/ && chmod 600 ~/.ssh/nl2cuda_gpu`（Windows 侧免拷）。",
     "与 `python skill/scripts/bench_case.py --case <name>`（前反向各 ≥1.05× torch.compile 才达标）。"),
    # ---- README.md（项目对外文档，一条加固 bullet 提了作者脚本）----
    ("README.md",
     "- **评测鲁棒性加固**：`run_on_a100.sh --auto-scale` 自适应放大到\"计算主导区\"（补短核假象）+ 规模敏感复测（擦线 PASS 放大掉破 1.05 判 `PASS_SCALE_SUSPECT`）+ `check_reference.py` 静态扫危险写法（补弱 baseline + kernel 嵌套重算 + 直调厂商库成品）+ 防作弊红线 §1-5 + 擦线 3 连稳定判据 + 驱动器健壮性（tar 自动回退 System32 bsdtar 绕杀软拦截 + 远程执行墙钟超时防 WSL 冻结）。",
     "- **评测鲁棒性加固**：自测应选\"计算主导区\"规模（baseline ≥1ms）避短核假象 + 擦线 PASS 换 ×2/×4 规模复测防规模挑选 + `check_reference.py` 静态扫危险写法（补弱 baseline + kernel 嵌套重算 + 直调厂商库成品）+ 防作弊红线 §1-5 + 擦线 3 连稳定判据。"),
    # ---- CLAUDE.md ----
    ("CLAUDE.md",
     "4. **自测（自动，无需用户提）**：跑 `bash skill/scripts/run_on_a100.sh",
     "4. " + GENERIC_SELFTEST),
    ("CLAUDE.md",
     "`run_on_a100.sh` 经双跳 SSH 直传远程 A100 跑评测（Windows 侧密钥已就绪；WSL 需先拷密钥）。详见 `AGENTS.md` 与 `skill/AUTONOMOUS_LOOP.md`。",
     "GPU 环境准备（Colab / 本地 GPU / 远程 SSH）见 `USAGE.md`。自测用 `verify_case.py` / `bench_case.py`。"),
    # ---- CONVENTIONS.md ----
    ("CONVENTIONS.md",
     "4. **自测（自动）**：已配 `--auto-test`，你每次改完会自动跑 `run_on_a100.sh",
     "4. " + GENERIC_SELFTEST),
    ("CONVENTIONS.md",
     "> 完整流程与 GPU 自测环境说明见 `AGENTS.md` 与 `skill/AUTONOMOUS_LOOP.md`。",
     "> 完整流程见 `AGENTS.md`；GPU 环境准备（Colab / 本地 / 远程 SSH）与自测方式见 `USAGE.md`。"),
    # ---- skill/SKILL.md（行内 run_on_a100 便利提及）----
    ("skill/SKILL.md",
     "命中则打印 WARN 供核查（有合法例外如变系数递推 T²，故只警告不拦）。`run_on_a100.sh` 每次自测前也会自动跑此预检并把 WARN 打进日志",
     "命中则打印 WARN 供核查（有合法例外如变系数递推 T²，故只警告不拦）。建议每次自测前先跑一遍此预检"),
    ("skill/SKILL.md",
     "`run_on_a100.sh` 会自动探测短核并放大规模重测（harness 兜底，无需你建 bench.env）**——所以务必让 config 参数化规模",
     "自测时应**手动选一个计算主导区规模**（baseline ≥1ms）重测确认，别被短核规模的虚高加速比骗**——所以务必让 config 参数化规模"),
    ("skill/SKILL.md",
     "，是**规模挑选**。`run_on_a100.sh` 已自动检测：擦线 PASS+短核时自动 ×2/×4 复测，掉破 1.05 判 `VERDICT=PASS_SCALE_SUSPECT`（strict 下算未达标，逼你优化 kernel 本体而非挑规模）。**对策**：选规模让 baseline 进计算主导区（≥1ms）再优化；bench.env 声明的规模要经得起放大交叉验证。",
     "，是**规模挑选**（不诚实）。**对策**：固定一个计算主导区规模（baseline ≥1ms）把 kernel 优化到达标；擦线 PASS 时自己换 ×2/×4 规模复测，若掉破 1.05 说明是短核固定开销虚高、非真达标，须优化 kernel 本体而非挑规模。"),
    # ---- skill/loop.md ----
    ("skill/loop.md",
     "固定一个计算主导区规模（baseline ≥1ms）优化到达标。`run_on_a100.sh` 会对擦线 PASS+短核自动 ×2/×4 复测，掉破 1.05 判 `PASS_SCALE_SUSPECT`（strict 下未达标）——挑更大规模也过不了，只能真优化。见 SKILL.md 达标判据\"规模挑选\"。—— 实测：GroupNorm gptme 挑 GN_N=384(baseline 前0.92ms)前向擦线1.07，放大768/1536 掉1.048/1.035 被拆穿。",
     "固定一个计算主导区规模（baseline ≥1ms）优化到达标。擦线 PASS 时自己换 ×2/×4 规模复测：若掉破 1.05 说明加速比来自短核固定开销摊薄、非 kernel 真更快，挑更大规模也过不了，只能真优化。见 SKILL.md 达标判据\"规模挑选\"。"),
    # ---- dead-link 清理：交付物不含内部测试记录(AGENT_TEST_MATRIX/CASE_EVIDENCE)，去掉指向它们的悬空引用 ----
    ("skill/SKILL.md",
     "> 上表各手段的**逐 case 实测加速比**（哪个 case 证了哪条、达到多少）见 [CASE_EVIDENCE.md](CASE_EVIDENCE.md)（三宿主测试时该附录会被剥离，避免开卷考）。",
     "> 上表各手段的通用性对**任意算法**成立——先把 kernel 归到某类瓶颈，再上对应手段。"),
    ("skill/SKILL.md",
     "> （各区间的**具体 case + 实测加速比**见 [CASE_EVIDENCE.md](CASE_EVIDENCE.md)，三宿主测试时剥离。）",
     "> （据此五类先归类你的算法，再定优化预期与策略。）"),
    ("skill/SKILL.md",
     "**判达标必须计算主导区多规模验证**（具体翻车实测见 [CASE_EVIDENCE.md](CASE_EVIDENCE.md)）。",
     "**判达标必须计算主导区多规模验证**。"),
    # ---- README.md：去掉指向内部测试矩阵的悬空引用 ----
    ("README.md",
     "。三宿主（aider/gptme/codex）逐形态对照与结论见 [`skill/AGENT_TEST_MATRIX.md`](skill/AGENT_TEST_MATRIX.md)。",
     "。均以纯自然语言输入在多个宿主 agent 上验证过 skill 的通用性。"),
]

# 兜底：改写后交付物里绝不允许残留的作者私有自测词。
FORBIDDEN = [
    r"run_on_a100", r"AUTONOMOUS_LOOP", r"--sync-cli", r"--round-cap",
    r"nl2cuda_gpu", r"双跳", r"ROUND_CAP_EXCEEDED", r"--auto-test",
    r"\.aider\.conf", r"start_gptme", r"prepare_cleanroom", r"11\.91\.", r"11\.127\.",
    # 内部测试记录/附录：交付物不含，去掉指向它们的悬空引用
    r"AGENT_TEST_MATRIX", r"CASE_EVIDENCE", r"MULTIAGENT_TEST_RESULTS",
]


def sanitize(dest_root):
    errors = []
    # 按文件分组规则
    by_file = {}
    for rel, anchor, repl in LINE_RULES:
        by_file.setdefault(rel, []).append((anchor, repl))

    for rel, rules in by_file.items():
        path = os.path.join(dest_root, rel)
        if not os.path.exists(path):
            errors.append(f"目标文件不存在（清单可能漏含）: {rel}")
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for anchor, repl in rules:
            hit = False
            for i, line in enumerate(lines):
                if line is None:          # 已被前一条规则标记删除，跳过
                    continue
                if anchor in line:
                    hit = True
                    if repl is None:
                        lines[i] = None  # 标记删除
                    else:
                        # 保留行尾换行
                        nl = "\n" if line.endswith("\n") else ""
                        lines[i] = repl + nl
                    break
            if not hit:
                errors.append(f"[drift] {rel}: 锚点未命中（主仓正文可能已改，规则过期）: {anchor[:40]}...")
        lines = [l for l in lines if l is not None]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    if errors:
        print("[sanitize] ✗ 规则失效（drift）——请更新 _sanitize_for_delivery.py 的规则：", file=sys.stderr)
        for e in errors:
            print("   " + e, file=sys.stderr)
        return 1

    # 兜底扫描：交付物全目录（.md 文件）不得残留禁用词
    leaks = []
    for root, _, files in os.walk(dest_root):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8") as f:
                txt = f.read()
            for pat in FORBIDDEN:
                if re.search(pat, txt):
                    leaks.append(f"{os.path.relpath(p, dest_root)} 含禁用词: {pat}")
    if leaks:
        print("[sanitize] ✗ 兜底扫描发现残留作者私有自测词：", file=sys.stderr)
        for l in leaks:
            print("   " + l, file=sys.stderr)
        return 1

    print("[sanitize] ✓ 方法论/约定文件已洗净（自测步→通用 verify/bench，无 run_on_a100/AUTONOMOUS_LOOP 等残留）")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python _sanitize_for_delivery.py <交付物根目录>", file=sys.stderr)
        sys.exit(2)
    sys.exit(sanitize(sys.argv[1]))
