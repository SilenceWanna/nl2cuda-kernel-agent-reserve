# USAGE.md — 从零到 `.cu`：用本 skill 把自然语言算法变成 CUDA 前反向 kernel

本说明书带你从**拉取交付物**一路走到**产出一个可独立编译的 `.cu` 文件**。全程只需照步骤操作，无需预先理解内部实现。

> **本 skill 做什么**：你用自然语言描述一个算法（+ 输入 shape/dtype + 对哪些输入求梯度），
> 驱动一个 AI 编码 agent（Claude Code / Codex / aider / Cursor 等任意能读文档+执行 shell+访问 GPU 的 agent），
> 自动写出 PyTorch 参考实现 + 手写 CUDA 前向&反向 kernel，以 PyTorch 为金标准验证正确性，
> 并在规范计时下力争超过 `torch.compile`。最终交付一个不依赖 PyTorch、可独立编译的 `.cu`。

---

## 步骤 0：获取交付物

本项目是 public 仓库，直接 clone（无需认证）：

```bash
git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git
cd nl2cuda-kernel-agent
```

**交付物包含的核心文件**（clone 后即全部就绪，无需挑选）：

| 路径 | 作用 | 你会不会改 |
|------|------|-----------|
| `framework/` | 算法无关的评测基座（正确性验证 + 计时 + CUDA 编译加载） | **只读，勿改** |
| `skill/SKILL.md` | 方法论主体（agent 读它执行全流程） | 只读 |
| `skill/DESIGN.md` / `loop.md` | 架构 / 优化迭代循环 | 只读 |
| `skill/scripts/verify_case.py` | 正确性验证 CLI | 只读 |
| `skill/scripts/bench_case.py` | 计时 CLI（vs torch.compile） | 只读 |
| `skill/scripts/profile_case.py` | 瓶颈诊断 CLI（优化时用） | 只读 |
| `skill/scripts/check_reference.py` | reference 静态预检（防弱 baseline） | 只读 |
| `cases/rbf/` | 唯一内置完整样例（agent 复制它作结构模板；含 `delivery/` 独立编译成品样例） | 参考 |
| `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md` | agent 启动约定（不同宿主自动读对应文件） | 只读 |
| `requirements.txt` / `scripts/probe_env.py` / `notebooks/run.ipynb` | 依赖 / 环境探测 / Colab 入口 | 只读 |

> **你的产物只落在新建的 `cases/<你的算法名>/` 目录**，framework 与上述工具都不动。

---

## 步骤 1：准备 GPU 环境（并冒烟验证内置 rbf 样例）

编译运行 CUDA kernel **需要 NVIDIA GPU + nvcc + PyTorch(cuda) + ninja**。按你的情况选一条路径。**每条路径都以"跑通内置 rbf 样例"收尾**——rbf 一 PASS，就证明环境就绪，可以进步骤 2。

### 路径 A（推荐，零成本）：Google Colab 免费 T4

1. 打开 [Colab](https://colab.research.google.com/)，`代码执行程序 → 更改运行时类型 → T4 GPU`。
2. 上传并打开本仓库的 `notebooks/run.ipynb`。
3. **从上到下依次运行它的单元即可**——notebook 已包含：clone/pull 仓库 → `pip install ninja` → `probe_env.py`（确认 GPU/CUDA/torch）→ `smoke_test.py`（确认 nvcc+ninja 编译链路）→ `verify_case.py --case rbf`（正确性全 PASS）→ `bench_case.py --case rbf`（加速比）。
4. 跑到 rbf 的 `verify` 全 PASS，环境即就绪。**走完 notebook = 环境准备 + 冒烟验证都已完成**，直接进步骤 2。

> **⚠️ Colab 运行时会重置**：闲置约 90 分钟或断网后，Colab 回收运行时——**clone 的仓库、上传/解包的 case 文件、已编译的 kernel 全部丢失**。回到 Colab 若发现命令报"找不到文件/目录"，先 `!pwd && ls cases/` 确认：仓库没了就**重跑开头的 clone+cd 单元恢复**，再重新放入你的 case。
> **⚠️ 本地 agent + Colab 只能"半自动"**：你本地跑的 CLI agent（aider/gptme 等）进程**碰不到 Colab 云端 GPU**——只能"agent 在本地产码 → 你手动把 `cases/<name>/` 搬进 Colab（如 git 中转、或打包/解包）跑 `verify/bench` → 把结果贴回给 agent 迭代"。agent 无法在 Colab 里自主自测。要 agent **全自主**自测，需 agent 与 GPU 同环境：路径 B（本机有卡）、路径 C（SSH 到有卡机），或让 agent 本身运行在 Colab 内。

**半自动：把本地 agent 产的 case 搬进 Colab（打包/解包，无需 git 认证）**

1. 本地在 case 目录的**上一级**（含 `cases/` 那层）打包成一行 base64（复制它）：
   - **Linux / macOS / WSL / Git Bash**：
     ```bash
     tar czf case.tgz --exclude='__pycache__' --exclude='*.pyc' cases/<你的算法名>
     base64 -w0 case.tgz          # 输出一整行 base64
     ```
   - **Windows PowerShell**（`base64 -w0` 在 PowerShell 不可用；tar 是 Win10/11 自带的）：
     ```powershell
     tar czf case.tgz --exclude=__pycache__ --exclude=*.pyc cases/<你的算法名>
     [Convert]::ToBase64String([IO.File]::ReadAllBytes("case.tgz")) | Set-Content -NoNewline case_b64.txt
     # 打开 case_b64.txt 复制那一整行（别用 certutil -encode——它带页眉/多行，需手工清理易错）
     ```
2. Colab 里新建 cell 解包（把上面那行 base64 粘进 `B64="..."`，**必须完整、结尾通常是 `==`**）：
   ```python
   import base64, tarfile, io
   B64 = "在此粘贴完整的一行 base64"
   with tarfile.open(fileobj=io.BytesIO(base64.b64decode(B64))) as t:
       t.extractall(".")     # 注意：解到当前目录，确保当前 cwd 在仓库根
   !ls cases/<你的算法名>/kernels/
   ```
3. Colab 里新建 cell 跑自测（**下面这段可直接复制**，把 `<你的算法名>` 换成实际名如 `rmsnorm`；`!` 前缀已带好）：
   ```python
   !python skill/scripts/verify_case.py --case <你的算法名>
   !python skill/scripts/bench_case.py  --case <你的算法名>
   ```
   若 `bench` 报"短核假象警告"（baseline <1ms），用大规模复测（`<规模ENV>` 见该 case 的 `config.py`，如 RMSNorm 是 `RMS_B`）：
   ```python
   !<规模ENV>=262144 python skill/scripts/bench_case.py --case <你的算法名>
   ```
   改一版就重打包解包一次（回第 1-2 步）。

> **⚠️ Colab code cell 是 Python**：跑命令行工具**必须行首加 `!`**（上面已带）。若你从别处（如本文档步骤 3 的通用命令块）复制不带 `!` 的命令，会被当 Python 解析报 `SyntaxError`——在 Colab 里每行前补 `!`。
> **⚠️ `<...>` 是占位符，别照抄尖括号**：`--case <你的算法名>` 要替换成实际名如 `--case rmsnorm`，不是原样敲 `--case <rmsnorm>`（`<` 会被 shell 当重定向符报错）。
> **⚠️ 每次跑命令前确认 cwd 在仓库根**：Colab cell 间 cwd 可能漂移或运行时重置，命令报"找不到文件"时先 `%cd /content/<仓库名>` 再跑。
> **⚠️ Colab 免费额度有限**：连续用一段时间会限额（提示无可用运行时/需等待或升级）。限额后转路径 B（自备/云 GPU）或路径 C（SSH 远程 GPU）继续；case 已在本地，同样打包搬过去即可。

### 路径 B：自备本地 / 云 GPU 机

> **Windows 用户请在 WSL（Ubuntu）里操作**——CUDA 编译链路在 WSL 下最顺（宿主装好 NVIDIA 驱动，WSL 内装 CUDA toolkit + PyTorch(cuda)）。在 PowerShell 里直接跑通常会卡在 nvcc/编译环节。进入 WSL：开始菜单搜 "Ubuntu" 或终端里 `wsl`。

在 GPU 机（Windows 则在 WSL）里：
```bash
git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git && cd nl2cuda-kernel-agent
pip install -r requirements.txt          # torch>=2.4, numpy>=1.26, ninja>=1.11
nvcc --version                           # 确认 nvcc 在 PATH
export CUDA_VISIBLE_DEVICES=0             # 指向一张空闲卡
python scripts/probe_env.py              # 确认 GPU / CUDA / PyTorch
python framework/smoke_test.py           # 确认编译链路（nvcc + ninja）
python skill/scripts/verify_case.py --case rbf   # 冒烟：前反向 5 种子应全 PASS
python skill/scripts/bench_case.py  --case rbf   # 冒烟：加速比（rbf 参考前~1.10×/反~1.17×）
```
rbf `verify` 全 PASS 即环境就绪，进步骤 2。

### 路径 C：本地无 GPU、SSH 连远程 GPU 机

很常见（本地开发机无卡，用远程/云 GPU；甚至需经跳板机中转两跳）。下面以「本地 Windows、经跳板机两跳到远程 GPU 机」为例（单跳更简单，去掉 ProxyJump 即可）。

**1. 配 SSH 别名（把多跳收成一个别名，一步登入）**

在 `~/.ssh/config` 里配（**Windows 放 `C:\Users\你的用户名\.ssh\config`**；Linux/WSL 放 `~/.ssh/config`）：
```sshconfig
Host gpubox                       # 远程 GPU 机（最终目标）
    HostName 10.0.0.2             # 换成你的 GPU 机地址
    User youruser
    IdentityFile ~/.ssh/your_key
    IdentitiesOnly yes
    ProxyJump jump                # 若无跳板机，删掉这一行
Host jump                         # 跳板机（无则整段删）
    HostName 10.0.0.1
    User jumpuser
    IdentityFile ~/.ssh/your_key
```
密钥自行管理、**不要提交进任何仓库**。验证一步登入 + 挑一张空闲卡（记下 `memory.used` 最小的那号，下面要用）：
```bash
ssh gpubox "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
```

**2. 远程机备环境**：在远程机上按路径 B 装好 CUDA toolkit + PyTorch(cuda) + ninja。

**3. 同步代码到远程**（本地已 clone 交付仓，在其根目录下操作）：
- **Windows（用自带的 `scp`，经上面别名同样两跳直达）**：
  ```powershell
  ssh gpubox "mkdir -p ~/nl2cuda"
  scp -r framework skill cases scripts notebooks requirements.txt AGENTS.md CLAUDE.md CONVENTIONS.md README.md USAGE.md gpubox:~/nl2cuda/
  ```
- **Linux / WSL（用 `rsync`）**：`rsync -az --exclude='.git' ./ gpubox:~/nl2cuda/`
- 以后每次改完 `cases/<name>/` 都同步过去再跑。

**4. 在远程跑冒烟验证**（⚠️ 三个易踩点，照做即可）：
- **非交互 ssh 不加载 shell profile**，命令里要**显式 `export PATH`** 指向远程的 conda/CUDA，否则找不到 python/nvcc；
- **必须显式指定空闲卡号**（把下面的 `0` 换成第 1 步挑的号，直接写数字，别写成 `<0>`）；
- **Windows PowerShell 里 `$` 要写成 `` `$ ``**（反引号转义）才能原样传给远程 bash（Linux/WSL 下去掉反引号）。
  ```powershell
  # Windows PowerShell（远程路径按你的实际 conda/CUDA 位置改）：
  ssh gpubox "export PATH=`$HOME/miniconda3/bin:/usr/local/cuda/bin:`$PATH; cd ~/nl2cuda && CUDA_VISIBLE_DEVICES=0 python framework/smoke_test.py && CUDA_VISIBLE_DEVICES=0 python skill/scripts/verify_case.py --case rbf"
  ```
  > 嫌引号/转义烦：把这串远程命令写进一个 `run.sh` 放到远程，本地只 `ssh gpubox "bash ~/run.sh"`，躲开引号地狱。
- 首次会编译 rbf kernel（+ torch.compile baseline），要等几分钟、中途无输出正常，别当卡死。

**5. rbf `verify` 全 PASS 即就绪。** 步骤 2 起，让 agent 按同样的"同步→远程跑→读结果"方式自测你的新算法——告诉 agent 你的别名（如 `ssh gpubox` 一步可登）、同步命令、以及远程跑命令要带的 `export PATH` 和 `CUDA_VISIBLE_DEVICES=你的卡号`。

> **⚠️ 有些 GPU 机默认不暴露 GPU**（`torch.cuda.is_available()` 为 False，须显式 `CUDA_VISIBLE_DEVICES=<卡>` 才可见）。**且 agent 的自动测试机制（若它每次改完自动跑一条固定测试命令）的子进程未必继承你 shell 里 `export` 的环境变量**——所以**把 `CUDA_VISIBLE_DEVICES=<卡> CUDA_ARCHS=<架构>` 直接写进自测命令每条前缀**，别只靠 shell export，例如测试命令设为 `CUDA_VISIBLE_DEVICES=0 CUDA_ARCHS=80 python skill/scripts/verify_case.py --case <名> && CUDA_VISIBLE_DEVICES=0 CUDA_ARCHS=80 python skill/scripts/bench_case.py --case <名>`。

> `framework/loader.py` 默认为 sm_75(T4) + sm_80(A100) 编译。其他架构用 `export CUDA_ARCHS="架构号"`（如 A100 用 `80`、H100 用 `90`）。

---

## 步骤 2：告诉 agent 你的 GPU 环境 + 输入算法描述

交付物已随仓库带了 agent 自动加载的约定文件（`AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md`）——**多数情况下你不用贴长提示词，直接告诉 agent 两件事即可**：

```
我的 GPU 环境是：<Colab / 本地WSL GPU / 远程SSH（把连接方式给它，如 "ssh gpubox" 一步可登）>。

<然后直接用自然语言描述你的算法>
例如："把每行向量归一化到均值0方差1再乘可学习的 gamma 加 beta，输入是 [批, 特征] 的 fp32 张量，对输入和 gamma/beta 都要能求梯度。"
```

agent 会（按 `SKILL.md` 流程）：
- 若你的描述是模糊自然语言 / 点了个有多变体的算子名 → **先推导数学规格 + PyTorch reference 呈给你确认**（唯一的人类确认点），确认后才动工；
- 据你说的环境类型，在对应位置（Colab 单元 / 本地 WSL / 远程 SSH）跑自测；
- 自建 `cases/<name>/`（reference / config / __init__ / op / kernels），自主推导反向。

> **兜底**：若你的 agent 没自动加载约定文件（或想更明确），把这段贴给它——
> ```
> 读 skill/SKILL.md 与 skill/DESIGN.md，严格按其流程和防作弊红线执行；framework/ 只读，cases/rbf/ 是结构范例。
> 我的 GPU 环境：<...>。我的算法：<自然语言描述 + 若有则给 shape/dtype + 对哪些输入求梯度>。
> 按 SKILL 步骤 0.5 先推导数学规格呈我确认，再建 cases/<name>/、写前反向 kernel、自测到达标。
> 红线：fp32、不用 fast-math、不改 framework/、cuBLAS/cuSOLVER 只作积木不得直调等价库成品(如分解直调 potrf)。
> ```

---

## 步骤 3：agent 自主自测到达标（你只在数学确认点介入）

确认数学规格后，agent 应**自主循环**：写 kernel → 在你的环境跑 `verify_case.py`（正确性）→ 通过后 `bench_case.py`（性能）→ 未达标按 `loop.md` 优化 → 重跑，直到达标或诚实报边界。你通常无需手动跑命令。

若想自己核验（或 agent 需要你代跑），这几条只读 CLI（在步骤 1 选定的环境里跑；**在 Colab 里每行前加 `!`**，本地/远程 shell 不加）：
```bash
python skill/scripts/verify_case.py    --case <你的算法名>   # ① 正确性 allclose，须全 PASS（不过先修正确性、别看性能）
python skill/scripts/check_reference.py --case <你的算法名>   # ② reference 静态预检，期望 REF_CHECK=CLEAN（防弱 baseline 假象）
python skill/scripts/bench_case.py     --case <你的算法名>   # ③ 性能，前反向各自 ≥1.05× torch.compile 才达标
python skill/scripts/profile_case.py   --case <你的算法名>   # ④ 未达标时诊断瓶颈（据结果按 loop.md 选手段）
```

- **达标线**：前向、反向各自 ≥1.05× `torch.compile`；擦线（1.05–1.10×）建议连跑 3 次都过。
- **⚠️ 短核假象（务必看）**：若 `bench_case.py` 报的 baseline 前向或反向 **<1ms**，加速比被固定开销（kernel launch/同步）抬高、**不可信、不算达标**——`bench_case.py` 会打印"短核假象警告"提示你放大规模。**小 reduce / 归一化（LayerNorm/RMSNorm 类）/ 逐元素算子尤其易中招**：默认小规模常报虚高的"1.5×~1.9× PASS"，放大到计算主导区（baseline≥1ms）后常回落到 <1.05 甚至 <1（带宽墙）。**做法**：`config.py` 让规模支持 env 覆盖（如 `RMS_B`），用大规模复测——`RMS_B=262144 python skill/scripts/bench_case.py --case <名>`；再换 2× 规模看加速比稳不稳（真优势稳、短核假象掉）。**别拿默认小规模的高加速比当达标。**
- **本征边界**：有些算法（纯访存前向、cuSOLVER/cuFFT 级厂商库前向）手写打不过是正常的——`SKILL.md` 的"识别本征边界"章节讲了何时该停、诚实报边界，别空转。

> **agent 能否自主放大复测？** 有 shell 执行能力的 agent（能自己跑命令）可自主放大；但**纯 API 会话式 agent（只会跑一条固定的自动测试命令、不能自己改规模）**——这类要么把大规模写进那条测试命令（如让它跑 `RMS_B=262144 python skill/scripts/bench_case.py --case <名>`），要么你手动跑大规模命令、把输出贴回给 agent 判断（半自动）。

---

## 步骤 4：产出并整理 `.cu` 交付

达标（或确认边界）后，你的 CUDA kernel 就在：

```
cases/<你的算法名>/kernels/*.cu
```

这已是可用的 `.cu`。若要一份**不依赖 PyTorch、可独立 `nvcc` 编译**的交付版（像 `cases/rbf/delivery/`），
照该样例的结构整理（可让 agent 做）：

| `cases/rbf/delivery/` 里的文件 | 作用 |
|------|------|
| `rbf_kernels.cu` | 前向+反向 `__global__` kernel + `extern "C"` 裸指针 host 接口（内部管理 device 内存） |
| `rbf_test.cu` | 自测 harness：内置 CPU 参考对拍 GPU 输出，`allclose` 打印 PASS/FAIL（不依赖 torch） |
| `Makefile` | `make test` 一键 nvcc 独立编译 + 跑自测；`make ARCH=sm_75 test` 换架构 |
| `README.md` | 算法说明 + 接口 + 编译运行方式 + 合规声明 |

独立交付版验证：
```bash
cd cases/<你的算法名>/delivery
make test        # 编译 + CPU 对拍，应打印 PASS
```

**至此你获得了最终的 `.cu`**（`kernels/*.cu` 或 `delivery/*_kernels.cu`），可嵌入你自己的 C++/CUDA 项目。

---

## 常见坑

- **`ninja: command not found`**：`pip install ninja`（Colab 默认没装；torch 的 JIT 编译后端必需）。
- **`需要 CUDA GPU`**：当前环境无 GPU 或 `CUDA_VISIBLE_DEVICES` 没指对卡。
- **架构不匹配**：新卡（如 H100 sm_90）需 `export CUDA_ARCHS="90"`。
- **短核假象**：若 `bench_case.py` 显示 baseline 前/反向 <0.15ms 却给出 1.2×+ 高加速比，多半是固定开销虚高、不是真达标。把 `config.py` 的规模调大（进入 baseline ≥1ms 的"计算主导区"）重测才可信。
- **弱 baseline 假象**：`check_reference.py` 报 WARN 时先核查——reference 若用了 Python for 循环遍历张量维度、O(T²) 密集矩阵伪向量化、或按规模切实现分支，会让 baseline 畸形慢、加速比虚高不诚实。reference 必须是单一、最干净的向量化写法。
- **红线**：全程 fp32、不用 fast-math、不改 `framework/`；cuBLAS/cuSOLVER 只能当积木（GEMM/TRSM/scan）自己拼算法，不得直调与目标算子等价的库成品（如 Cholesky 直调 `potrf`）。

---

## 一句话流程

**clone 仓库 → 备好 GPU 环境（Colab / 本地 WSL / 远程 SSH，任一路径都以跑通 rbf 收尾）→ 告诉 agent 你的环境类型 + 用自然语言描述算法 → agent 推导数学规格请你确认 → agent 自建 `cases/<name>/` 并自主 verify/bench 自测到达标 → 从 `kernels/*.cu` 取得 `.cu`（可选整理成 `delivery/` 独立编译版）。**
