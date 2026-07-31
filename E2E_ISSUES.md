# 端到端测试问题记录（E2E_ISSUES）

> 本文档记录**用户按 `USAGE.md` 从零走到产出 `.cu`** 的端到端测试中暴露的问题。
> 每个问题记：现象 / 期望 / 影响范围 / 状态。**解决一个就更新其状态与方案**；后续出现新问题也追加到这里。
>
> 状态图例：🔴 待解决 · 🟡 进行中 · ✅ 已解决（附 commit/方案）

**最终目标校准**：本项目的终态是——用户**只输入自然语言算法描述**，agent 依 skill 预设方法**自动选定并连接 GPU、建 case、写前反向 kernel、自测到达标、产出 `.cu`**，全程无需用户懂内部实现、无需贴长提示词、无需手动配 GPU 自测。下列问题多数是"离这个终态还差什么"。

## 架构决策（2026-07-30，处理 #5 时定）

**双仓分离**：
- **当前 git 仓（本仓）= 作者的开发/测试仓**——保留全部自测基础设施（`run_on_a100.sh`/`autotest.sh`/`AUTONOMOUS_LOOP.md`/京东双跳 SSH/内部测试记录等），正文不动，作者继续用它自主调试与改进 skill。
- **`../nl2cuda-delivery/`（同级目录）= 最终交付版**——由 `make_delivery.sh` 从本仓生成：只含通用用户需要的文件，并经 `skill/scripts/_sanitize_for_delivery.py` 把方法论/约定文件里的作者私有自测内容**洗成通用 `verify_case`/`bench_case` 表述**。这才是发给用户、保证一般性的版本。
- **后续端到端测试用交付物目录**；若涉及 skill 本身的改动，同步回本仓（源），并在本文件记录。**改了本仓方法论正文后，务必重跑 `make_delivery.sh` 重新生成+净化交付物**（净化脚本带 drift 检测，正文改动导致锚点失效会 exit 1 提示更新规则）。

---

---

## 记录轮次

- **第 1 轮（2026-07-30，首次评审 USAGE.md 初稿）**：用户通读 USAGE.md 初稿，提出 6 个问题（下方 #1–#6）。尚未真正跑端到端，属"说明书评审"阶段暴露的问题。
- **第 2 轮（2026-07-31，用户亲自按交付仓 USAGE 实走路径 C）**：用户本地 Windows 无 GPU、经跳板机双跳连远程 A100，按路径 C 配置时暴露 3 个问题（下方 #7–#9）。属真正实走阶段的问题（前置：预跑已确认交付仓 rbf 在 A100 即装即用 verify/bench 全 PASS）。
- **第 3 轮（2026-07-31，准备 aider+gptme 端到端测试时厘清路径 A/B/C 执行模型）**：确定 gptme 走路径 A（半自动）、aider 在 A100 上装走路径 B。厘清中发现路径 A 对本地 CLI agent 只能半自动（#10）。

---

## 问题清单

### #1　自备 GPU 机步骤未说明在 WSL 中运行 ✅
- **现象**：`USAGE.md` 步骤 1 路径 B（自备 GPU 机）直接给 `pip install` / `export` / `python` 命令，未告知 Windows 用户这些应在 **WSL** 里跑。
- **期望**：路径 B 明确"Windows 用户在 WSL（Ubuntu）里操作"，给 WSL 进入方式与注意点（如 CUDA-on-WSL 需宿主装好驱动、WSL 内装 CUDA toolkit）。
- **影响**：Windows 本地有 GPU 的用户照初稿在 PowerShell 里跑会失败。
- **范围**：`USAGE.md` 步骤 1 路径 B。
- **状态**：✅ **已解决（2026-07-30）**：USAGE 路径 B 加 WSL 说明（Windows 用户进 WSL、宿主装驱动+WSL 装 CUDA toolkit、`wsl` 进入方式）。

### #2　步骤描述重复——路径 A 已含步骤 2 ✅
- **现象**：步骤 1 路径 A（Colab）让用户跑 `notebooks/run.ipynb`，而该 notebook 的单元**已经包含**步骤 2 的 `verify_case.py --case rbf` / `bench_case.py --case rbf`。于是走路径 A 的用户到步骤 2 会发现"已经做过了"，步骤划分重复、易困惑。
- **期望**：重新组织步骤——把"环境准备"与"冒烟验证"的关系理顺：路径 A 走完 notebook 即等于走完步骤 1+2；路径 B 才需手动跑步骤 2。或按"环境（A/B）→ 二者都以冒烟 rbf 收尾"重构，不让步骤 2 显得是新动作。
- **影响**：说明书逻辑不顺，用户重复操作/困惑。
- **范围**：`USAGE.md` 步骤 1、步骤 2 的划分。
- **状态**：✅ **已解决（2026-07-30）**：重构为"步骤 1 = 准备环境 + 冒烟 rbf"，每条路径（A/B/C）都以跑通 rbf 收尾；原独立的"步骤 2 冒烟"并入，后续步骤顺序号前移（原步骤 3/4/5 → 现 2/3/4）。

### #3　缺"远程 SSH 连接 GPU"场景的配置指导 ✅
- **现象**：`USAGE.md` 只给了 Colab / 本地自备 GPU 两条路径。但真实用户（如本项目作者自己）常是**本地无 GPU、需 SSH 连远程 GPU 机**（甚至经跳板机双跳）。初稿无此场景指导。
- **期望**：增第三条路径 C「远程 SSH GPU」——指导用户如何配置 SSH（含可能的跳板机双跳）、把代码同步到远程、在远程跑 verify/bench。**注意**：作者现有的 `run_on_a100.sh` 是私有封装（见 #5），不能直接给通用用户；需给的是"通用 SSH 连接与代码同步方法"，让 agent/用户据此自建，而非暴露作者的连接细节。
- **影响**：这类用户（无本地 GPU + 有远程 GPU）无从下手。
- **范围**：`USAGE.md` 步骤 1 新增路径 C。
- **状态**：✅ **已解决（2026-07-30）**：USAGE 新增路径 C「远程 SSH GPU」——通用做法（`~/.ssh/config` + ProxyJump 配跳板、远程装环境、rsync/scp/git 同步、`ssh <别名> '…verify/bench'`），只给通用方法框架、不含作者私有连接细节。SKILL 步骤 0 也纳入"远程 SSH：同步代码→远程跑→读结果"通用流程。

### #4　步骤 3 提示词过长——应能只输自然语言 ✅
- **现象**：`USAGE.md` 步骤 3 让用户把一段很长的上手提示（读 SKILL、建 7 文件、守红线…）贴给 agent。但阶段 7 内化的成果本应是——用户**只输一句算法自然语言**，agent 靠自动加载的约定文件（AGENTS/CLAUDE/CONVENTIONS）+ SKILL 自主走完，不需长提示词。
- **期望**：步骤 3 简化为"（约定文件已随交付物加载）直接输入你的算法自然语言描述即可"，长提示词最多作为"约定文件没生效时的兜底"。**从单纯自然语言直接生成 .cu 是本项目最终目标**，说明书应体现而非倒退。
- **影响**：说明书让交付物显得比实际更难用，掩盖了"内化"成果。
- **范围**：`USAGE.md` 步骤 3；依赖 AGENTS/CLAUDE/CONVENTIONS 在交付物里确实能被宿主自动加载。
- **状态**：✅ **已解决（2026-07-30）**：USAGE 步骤 2 改为"告诉 agent 环境类型 + 纯自然语言描述算法"（约定文件已随交付物加载，靠它 + SKILL 自主走完）；长提示词降级为"约定文件没生效时的兜底"。

### #5　交付物含作者私有自测内容（run_on_a100 等）——破坏一般性 🔴（重要）
- **现象**：`USAGE.md` 步骤 4 出现 `run_on_a100.sh`；更严重的是**方法论/约定文件正文里也大量嵌了作者自测脚本引用**。实测分布：
  - `skill/AUTONOMOUS_LOOP.md`：**34 处**（run_on_a100/VERDICT/--sync-cli/--round-cap 等，几乎通篇围绕作者的远程自测封装写）
  - `AGENTS.md`：6 处、`CLAUDE.md`：3 处、`CONVENTIONS.md`：2 处、`skill/SKILL.md`：3 处、`skill/loop.md`：2 处、`USAGE.md`：1 处
- **期望**：**交付物里任何文件都不应出现 `run_on_a100.sh`/`autotest.sh`/`start_gptme.sh`/`prepare_cleanroom.sh`/`--sync-cli`/`--round-cap`/京东双跳 SSH 等作者自测专用内容**。通用用户的自测是 `verify_case.py`/`bench_case.py`（+ 可选 `profile_case.py`/`check_reference.py`）。需把方法论/约定文件**净化**成宿主无关、环境无关的通用表述（自测=在你的 GPU 上跑 verify/bench，不预设"远程 A100 + 作者脚本"）。
- **影响**：**这是交付物一般性的核心短板**——通用用户拿到会被引导去跑一个不存在的私有脚本；AUTONOMOUS_LOOP.md 尤其严重（等于把作者的测试基建当成了方法论）。
- **范围**：`skill/AUTONOMOUS_LOOP.md`（重灾区，可能需大改或从交付物移除/重写）、`AGENTS.md`/`CLAUDE.md`/`CONVENTIONS.md`/`skill/SKILL.md`/`skill/loop.md`/`USAGE.md`。
- **状态**：✅ **已解决（2026-07-30，方案：双仓分离 + 交付物净化，见上方架构决策）**。
  - `make_delivery.sh` **从交付物排除** `AUTONOMOUS_LOOP.md`（它 100% 是作者远程自测玩法，非通用方法论；其方法论部分 VERDICT 决策/loop 纪律已在 `loop.md`）。
  - 新增 `skill/scripts/_sanitize_for_delivery.py`（本仓构建工具，**不进交付物**）：make_delivery 复制文件后调用它，把交付副本里 SKILL/loop/AGENTS/CLAUDE/CONVENTIONS/README 的作者私有自测引用（run_on_a100/双跳 SSH/--sync-cli/--round-cap/nl2cuda_gpu 等）就地洗成通用 `verify_case`/`bench_case` 表述。带 **drift 检测**（锚点失效即 exit 1）+ **兜底全目录禁用词扫描**（有残留即 exit 1）。
  - 独立复验：交付物全目录 grep 12 个作者私有自测词**零命中**；净化后文本通顺（自测步=在你的 GPU 上跑 verify/bench）。
  - 主仓正文**未改**（=作者测试仓，保留 run_on_a100 等）。USAGE.md 源修正：不再提 run_on_a100、文件清单去掉 AUTONOMOUS_LOOP.md。

### #6　步骤 4 自测应由 agent 自主完成（含自动连 GPU）✅
- **现象**：`USAGE.md` 把自测写成用户手动跑 CLI。但理想流程是——用户在贴算法描述**之前**先告诉 agent「我用 Colab / 本地 GPU / 远程 SSH」，agent 据 skill 里**预设好的方法**自动连接 GPU、建 case、写 kernel、跑自测、迭代到达标，用户不用手动跑命令。
- **期望**：
  1. `USAGE.md` 在输入算法描述前加一步「告知 agent 你的 GPU 环境（Colab / 本地 / 远程 SSH）」。
  2. **skill 里新增"agent 如何据环境类型自动连 GPU 并自测"的方法**（目前**尚无**这类方法——这是要新增的能力，非文档改写）。让 agent 读到环境类型后知道该怎么跑（Colab 里直接跑 / 本地直接跑 / 远程 SSH 按通用方式同步+跑）。
  3. 自测由 agent 自主循环（生成→verify→bench→按 loop.md 迭代），用户只在数学确认闸门介入。
- **影响**：当前需用户手动跑命令，未达"自然语言进、.cu 出"的自动化终态。
- **范围**：`USAGE.md` 步骤 3/4；**skill 新增环境自适应自测方法**（新能力）；与 #5 净化联动（新方法必须是通用的，不含作者私有脚本）。
- **状态**：✅ **已解决（2026-07-30）**：SKILL **步骤 0 新增"确认 GPU 环境 + 据环境类型准备自测"**——三类环境（Colab/本地WSL/远程SSH）各给 agent 的自测方式表 + 通用命令，born-general 无作者私有脚本（进交付物经净化脚本兜底扫描零残留）。USAGE 步骤 2 加"先告诉 agent 环境类型"、步骤 3 改为"agent 自主自测（你只在数学确认点介入）"。

### #7　步骤 0 clone 入口 URL 是旧仓名（改名后失效/指向错仓）🔴
- **现象**：交付仓 `USAGE.md` 步骤 0（及 README）的 `git clone` URL 仍是旧仓名 `nl2cuda-kernel-agent`。仓库已双仓分离改名——旧名现指向**开发/自测仓 reserve**（`nl2cuda-kernel-agent-reserve`，GitHub 对改名仓有重定向），而交付仓是 `nl2cuda-kernel-agent-skill`。用户照 USAGE 原样 clone 会拿到 reserve 仓（含作者私有内容），而非净化的 skill 交付仓。
- **期望**：交付仓里所有 clone/下载 URL 指向交付仓自身 `nl2cuda-kernel-agent-skill.git`。
- **范围**：`USAGE.md` 步骤 0、README.md、`notebooks/run.ipynb`（Colab clone 的 REPO_URL）、`skill/USING_WITH_OTHER_AGENTS.md`（若有 clone 指引）——凡交付物里出现的 clone URL。⚠️ 注意：这些 URL 在**开发仓源文件**里就是旧名（开发仓自己叫这个名合理），需在 `make_delivery.sh` 净化阶段把交付副本的 URL 改写成交付仓名（类似 run_on_a100 净化），否则每次重生又变回旧名。
- **状态**：✅ **已解决（2026-07-31）**：`make_delivery.sh` 复制后加全局改写步——`re.sub(r"nl2cuda-kernel-agent(?!-)", "nl2cuda-kernel-agent-skill", ...)`（负向断言不误伤已带 -skill/-reserve 后缀，幂等），把交付副本的 clone URL / REPO_URL / cd 目录名 / 标题全改成交付仓名。开发仓源保持旧名不变。独立验证：clone 交付仓后零裸旧名，所有 URL 指向 `nl2cuda-kernel-agent-skill`。

### #8　路径 C（远程 SSH）只给了 WSL/rsync 写法，未照顾 Windows 原生 driver 🔴
- **现象**：用户 driver 在 **Windows 原生**（不进 WSL），但 USAGE 路径 C 的代码同步给的是 `rsync -az ./ <别名>:~/...`（Windows 默认无 rsync）和 `tar czf -` 管道（Linux tar 写法）。Windows 用户无法直接照做。
- **期望**：路径 C 补 **Windows 原生**同步方式——`scp -r <文件列表> <别名>:~/dir/`（Windows 自带 OpenSSH 的 scp，经 `~/.ssh/config` 的 ProxyJump 同样两跳直达），或 Windows 自带 `tar.exe`(bsdtar) 打包。并说明 `~/.ssh/config` 在 Windows 放 `C:\Users\<你>\.ssh\config`。
- **范围**：`USAGE.md` 路径 C 的"代码同步"步。
- **状态**：✅ **已解决（2026-07-31）**：路径 C 重写为"以 Windows 两跳为主例"，代码同步给 **Windows `scp -r <文件列表>` + Linux/WSL `rsync`** 两版；补 `~/.ssh/config` 在 Windows 的位置（`C:\Users\你\.ssh\config`）、ProxyJump 双跳模板、一步登入+挑空闲卡验证。

### #9　远程命令里 `<占位符>` 尖括号被用户连着敲进去 → bash 语法错 ✅
- **现象**：USAGE 路径 C 的远程命令用 `CUDA_VISIBLE_DEVICES=<空闲卡号>` 这种尖括号占位符。用户照敲成 `CUDA_VISIBLE_DEVICES=<6>`，远程 bash 把 `<` 当重定向符号，报 `未预期的符号 '6' 附近有语法错误`。
- **期望**：占位符改用不会被 shell 误解析的写法——如给一个**可直接跑的具体示例**（`CUDA_VISIBLE_DEVICES=0`）+ 旁注"把 0 换成你的空闲卡号"，或用 `你的卡号` 中文占位（敲的人不会连中文一起留）。PowerShell 里远程命令的 `$` 转义（`` `$ ``）也补个提示，或建议"把远程命令写成 .sh 放远程、本地只 `ssh <别名> bash ~/run.sh`"躲开引号地狱。
- **范围**：`USAGE.md` 路径 C（及步骤 2/3 里所有含 `<...>` 占位的远程命令）。
- **状态**：✅ **已解决（2026-07-31）**：路径 C 的远程命令改用**可直接跑的具体示例**（`CUDA_VISIBLE_DEVICES=0` + 旁注"换成你挑的空闲卡号"），说明性占位改**中文**（`你的卡号`，敲的人不会连中文一起留）；并补 PowerShell `` `$ `` 转义提示 + "把远程命令写进 .sh 躲引号地狱"的替代法 + "非交互 ssh 须显式 export PATH"提示。

### #10　路径 A（Colab）对本地 CLI agent 只能半自动，USAGE 未说明 🔴
- **现象**：USAGE 路径 A 让用户"打开 Colab notebook 跑"，隐含 agent 能在 Colab 里自测。但 Colab GPU 在 Google 云端，**本地跑的 CLI agent（aider/gptme）进程碰不到它**——只能"本地 agent 产码 → 用户手动把 case 搬进 Colab 跑 verify/bench → 结果回贴给 agent 迭代"的**半自动**模式，agent 不自主自测。厘清 aider/gptme 端到端路径时发现（大多数用户若本地跑 agent + 用 Colab，都会撞上这个）。
- **期望**：USAGE 路径 A 明确两种子形态并给流程——**A-1 半自动**（本地 agent 产码 + 用户中转 Colab 跑 + 回贴，本地 CLI agent 的现实路径）；**A-2 全自动**（agent 本身运行在 Colab 环境内，如 Colab terminal 里驱动，才能自主摸 GPU）。并点明"agent 全自主自测需要 agent 与 GPU 同环境（路径 B 本机有卡 / 路径 C SSH 到有卡机 / 路径 A-2 agent 在 Colab 内）；本地 agent + Colab 只能半自动"。
- **范围**：`USAGE.md` 路径 A、步骤 3（自主自测的前提）。
- **状态**：🔴 待解决（第 3 轮记录，先记后修——本轮先做 aider/gptme 端到端，问题攒一起改）。

---

## 问题间的依赖与建议顺序

1. **先 #5（净化一般性）**——它是地基：AUTONOMOUS_LOOP.md 等把作者自测当方法论，不净化则 #4/#6 的"agent 自主自测"会继承私有脚本。
2. **再 #6 + #4**——在净化后的干净方法论上，新增"环境自适应自测"通用方法 + 简化为纯自然语言输入。
3. **同时 #1/#2/#3**——USAGE.md 的环境章节重构（WSL 说明、路径 A/B/C、步骤去重）。
4. 每解决一项，更新本文件对应状态为 ✅ 并附 commit。

**第 2 轮（#7–#9）建议顺序**：
1. **先 #7（clone URL 旧仓名）**——最重要：用户拿错仓等于净化全白费；须在 `make_delivery.sh` 净化阶段改写交付副本的 clone URL（否则每次重生又变回旧名），改完 `make_delivery.sh --push` 重推交付仓。
2. **再 #8 + #9**——USAGE 路径 C 补 Windows 原生同步（scp）+ 修占位符尖括号（改可直接跑的示例/中文占位）。二者都是 USAGE 文本改动，改完随交付物一起重推。
3. 都属"面向照着敲的新手不够友好"，本质是 USAGE 假设了 Linux/WSL + 熟悉占位符约定，需照顾 Windows 原生 driver。
