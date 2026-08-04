# USAGE.md — 从零到 `.cu`：用本 skill 把自然语言算法变成 CUDA 前反向 kernel

本说明书带你从**拉取交付物**一路走到**产出一个可独立编译的 `.cu` 文件**。全程只需照步骤操作，无需预先理解内部实现。

> **本 skill 做什么**：你用自然语言描述一个算法（+ 输入 shape/dtype + 对哪些输入求梯度），
> 驱动一个 AI 编码 agent（Claude Code / Codex / aider / Cursor 等任意能读文档+执行 shell+访问 GPU 的 agent），
> 自动写出 PyTorch 参考实现 + 手写 CUDA 前向&反向 kernel，以 PyTorch 为金标准验证正确性，
> 并在规范计时下力争超过 `torch.compile`。最终交付一个不依赖 PyTorch、可独立编译的 `.cu`。

## 先选你的路径

编译运行 CUDA kernel **需要 NVIDIA GPU + nvcc + PyTorch(cuda) + ninja**。按你有什么 GPU 选一条，然后**只看对应那一部分**：

| 你的情况 | 用哪部分 | 特点 |
|---------|---------|------|
| 没有 GPU，想零成本试 | **第一部分：路径 A（Google Colab）** | 最常用；agent 本地产码 + Colab 一次性验证（两阶段，规避 Colab 超时） |
| 有 GPU 机 / 能 SSH 到 GPU 机 | **第二部分：路径 B / C** | agent 直接在有卡环境自主自测；B=本机有卡，C=SSH 远程 |

两部分都先做下面这一步（获取交付物），然后各走各的。

---

## 步骤 0（两条路径共用）：获取交付物

本项目是 public 仓库，直接 clone（无需认证）：

```bash
git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent.git
cd nl2cuda-kernel-agent
```

**交付物核心文件**（clone 后即全部就绪）：

| 路径 | 作用 | 你会不会改 |
|------|------|-----------|
| `framework/` | 算法无关的评测基座（正确性验证 + 计时 + CUDA 编译加载） | **只读，勿改** |
| `skill/SKILL.md` / `DESIGN.md` / `loop.md` | 方法论主体 / 架构 / 优化迭代循环 | 只读 |
| `skill/scripts/{verify,bench,profile,check_reference}_case.py` | 验证 / 计时 / 诊断 / 预检 CLI | 只读 |
| `cases/rbf/` | 唯一内置完整样例（agent 复制它作结构模板；含 `delivery/` 独立编译成品样例） | 参考 |
| `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md` | agent 启动约定（不同宿主自动读对应文件） | 只读 |
| `requirements.txt` / `scripts/probe_env.py` | 依赖 / 环境探测 | 只读 |

> **你的产物只落在新建的 `cases/<你的算法名>/` 目录**，framework 与上述工具都不动。

---

# 第一部分 · 路径 A（Google Colab 免费 T4，最常用）

**Colab 的头号约束：运行时会超时/断连重置**（闲置约 90 分钟或断网），一旦重置，clone 的仓库、上传的 case、已编译的 kernel 全丢。而你本地跑的 AI agent 进程**碰不到 Colab 云端 GPU**。所以路径 A 用**两阶段**，把"易受超时影响的验证"压缩成 Colab 里的**一个 cell 一次性跑完**：

- **阶段一（本地，慢慢来、不碰 Colab）**：agent 在你本地把 case 写好（不受超时影响）。
- **阶段二（Colab，一个 cell 跑完）**：把 case 打包搬进 Colab，用一个 cell 从 clone 到 bench 一口气跑完。

## A-1　阶段一：让 agent 在本地产出 case

在本地（装好任意 AI 编码 agent，如 Claude Code / Codex / aider）打开 clone 下来的仓库，**直接用自然语言描述你的算法**——多数 agent 会自动读仓库里的约定文件（`AGENTS.md`/`CLAUDE.md`/`CONVENTIONS.md`），无需长提示词：

```
把每行向量归一化到均值0方差1再乘可学习的 gamma 加 beta，输入是 [批, 特征] 的 fp32
张量，对输入和 gamma/beta 都要能求梯度。
```

agent 会（按 `SKILL.md` 流程）：
- 若描述模糊 / 点了有多变体的算子名 → **先推导数学规格 + PyTorch reference 呈你确认**（唯一的人类确认点），确认后才动工；
- 自建 `cases/<你的算法名>/`（reference / config / __init__ / op / kernels），自主推导反向。

> **告诉 agent 你在 Colab 跑**：agent 本地无 GPU、跑不了自测，让它**只写代码不跑测试**（写完列出所有文件即可）。自测由你在阶段二做、把结果贴回给它迭代。
> **兜底**（agent 没自动读约定文件时，把这段给它）：
> ```
> 读 skill/SKILL.md 与 skill/DESIGN.md，严格按其流程和防作弊红线执行；framework/ 只读，cases/rbf/ 是结构范例。
> 我的算法：<自然语言描述 + 若有则给 shape/dtype + 对哪些输入求梯度>。我本地无 GPU（在 Colab 测），你只写代码不跑测试。
> 按 SKILL 步骤 0.5 先推导数学规格呈我确认，再建 cases/<name>/、写前反向 kernel。
> __init__.py 照 cases/rbf/__init__.py 直接 Case(...) 显式传全字段；config.py 让规模支持 env 覆盖。
> 红线：fp32、不用 fast-math、不改 framework/、cuBLAS/cuSOLVER 只作积木不得直调等价库成品(如分解直调 potrf)。
> ```

## A-2　打包 case（一行 base64）

产物稳定后，在仓库根目录（含 `cases/` 那层）打包成一行 base64：

- **Linux / macOS / WSL / Git Bash**：
  ```bash
  tar czf case.tgz --exclude='__pycache__' --exclude='*.pyc' cases/<你的算法名>
  base64 -w0 case.tgz          # 输出一整行 base64，复制它
  ```
- **Windows PowerShell**（`base64 -w0` 在 PowerShell 不可用；tar 是 Win10/11 自带的）：
  ```powershell
  tar czf case.tgz --exclude=__pycache__ --exclude=*.pyc cases/<你的算法名>
  [Convert]::ToBase64String([IO.File]::ReadAllBytes("case.tgz")) | Set-Content -NoNewline case_b64.txt
  # 打开 case_b64.txt 复制那一整行（别用 certutil -encode——它带页眉/多行，需手工清理易错）
  ```

## A-3　阶段二：Colab 一个 cell 从头跑到尾

1. 打开 [Colab](https://colab.research.google.com/)，`代码执行程序 → 更改运行时类型 → T4 GPU`。
2. **只用下面这一个 cell**（clone → 装依赖 → 解包 case → verify → bench 全绑一起，中间无停顿、不给超时留空隙）。把 `CASE` 改成你的 case 名、`B64` 粘上 A-2 那行 base64：
   ```python
   # 环境 + 解包 case + 自测，一个 cell 跑完（避免 Colab 分步被超时打断）
   import os, base64, tarfile, io
   os.chdir("/content")
   if not os.path.isdir("/content/nl2cuda-kernel-agent-skill"):
       !git clone https://github.com/SilenceWanna/nl2cuda-kernel-agent-skill.git
   os.chdir("/content/nl2cuda-kernel-agent-skill")
   !pip install ninja -q
   CASE = "你的算法名"                       # ← 改成实际 case 名，如 rmsnorm（不要留尖括号）
   B64  = "在此粘贴 A-2 那一整行 base64"      # ← 一整行，结尾通常是 ==
   with tarfile.open(fileobj=io.BytesIO(base64.b64decode(B64))) as t:
       t.extractall(".")
   print("case 文件:", os.listdir(f"cases/{CASE}"))
   !python skill/scripts/verify_case.py --case {CASE}
   !python skill/scripts/bench_case.py  --case {CASE}
   ```
   - `os.chdir`（Python，跨行不漂）比分散的 `%cd` 稳；`{CASE}` 是 f-string 插值，只填一处、不会残留尖括号。
   - 首次 nvcc 编译要等几分钟、中途无输出正常，别当卡死。
   - `verify` 全 PASS 才看 `bench`；正确性不过先让 agent 修正确性、别看性能。

3. **若 `bench` 报"短核假象警告"**（baseline <1ms，小 reduce/归一化/逐元素类常见，加速比被固定开销抬高、不可信），在同一运行时补一个 cell 放大规模复测（`<规模ENV>` 见该 case `config.py`，如 RMSNorm 是 `RMS_B`）：
   ```python
   !<规模ENV>=262144 python skill/scripts/bench_case.py --case 你的算法名
   ```
   再换 2× 规模看加速比稳不稳（真优势稳、短核假象会掉）。**别拿默认小规模的高加速比当达标。**

## A-4　迭代与达标

- 把 Colab 的 verify/bench 输出**贴回给本地 agent**，它据此改 kernel → 你重打包（A-2）→ **重跑 A-3 那一个 cell**（带 `if not isdir` 幂等，仓库在就跳过 clone、只重解包+自测）。
- **达标线**：前向、反向各自 ≥1.05× `torch.compile`；擦线（1.05–1.10×）建议连跑 3 次都过。
- **本征边界**：有些算法（纯访存前向如池化/逐元素、cuSOLVER/cuFFT 级厂商库前向）手写打不过是正常的——`SKILL.md` 的"识别本征边界"章节讲了何时该停、诚实报边界，别空转。

> **⚠️ Colab 免费额度有限**：连续用一段时间会限额（提示无可用运行时）。限额后转**第二部分**路径 B/C 继续——**产物已在本地，同样打包搬过去即可，不用重来**。

## A-5　取得 `.cu`

达标（或确认边界）后，你的 kernel 在 `cases/<你的算法名>/kernels/*.cu`。要独立编译交付版，见文末**取得 `.cu`**一节。

---

# 第二部分 · 路径 B / C（有 GPU 机 / SSH 远程，agent 自主自测）

这两条路径下 agent 与 GPU 同环境（真 shell），**agent 能自己跑命令、自主循环自测**，不用你手动搬运。二者**只在"环境准备"不同**，之后步骤全共用。

## 步骤 1：准备 GPU 环境并冒烟验证 rbf

先按你的情况配好环境，**都以"跑通内置 rbf 样例"收尾**——rbf 一 PASS 就证明环境就绪。

### 路径 B：自备本地 / 云 GPU 机（agent 与卡同机）

> **Windows 用户请在 WSL（Ubuntu）里操作**——CUDA 编译链路在 WSL 下最顺（宿主装好 NVIDIA 驱动，WSL 内装 CUDA toolkit + PyTorch(cuda)）。进入 WSL：开始菜单搜 "Ubuntu" 或终端 `wsl`。

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

### 路径 C：本地无 GPU、SSH 连远程 GPU 机

以「本地 Windows、经跳板机两跳到远程 GPU 机」为例（单跳更简单，去掉 ProxyJump 即可）。

**① 配 SSH 别名**（`~/.ssh/config`；**Windows 放 `C:\Users\你的用户名\.ssh\config`**，Linux/WSL 放 `~/.ssh/config`）：
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
密钥自行管理、**不要提交进任何仓库**。验证一步登入 + 挑空闲卡（记下 `memory.used` 最小的号）：
```bash
ssh gpubox "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
```

**② 远程机备环境**：在远程机上按路径 B 装好 CUDA toolkit + PyTorch(cuda) + ninja。

**③ 同步代码到远程**（本地已 clone，在其根目录）：
- **Windows（自带 `scp`，经别名同样两跳直达）**：
  ```powershell
  ssh gpubox "mkdir -p ~/nl2cuda"
  scp -r framework skill cases scripts requirements.txt AGENTS.md CLAUDE.md CONVENTIONS.md README.md USAGE.md gpubox:~/nl2cuda/
  ```
- **Linux / WSL（`rsync`）**：`rsync -az --exclude='.git' ./ gpubox:~/nl2cuda/`
- 以后每次改完 `cases/<name>/` 都同步过去再跑。

**④ 在远程跑冒烟**（⚠️ 三个易踩点）：
- 非交互 ssh 不加载 profile，命令里**显式 `export PATH`** 指向远程 conda/CUDA；
- **显式指定空闲卡号**（把 `0` 换成实际号，直接写数字，别写 `<0>`）；
- Windows PowerShell 里远程命令的 `$` 写成 `` `$ ``（反引号转义），Linux/WSL 去掉反引号：
  ```powershell
  ssh gpubox "export PATH=`$HOME/miniconda3/bin:/usr/local/cuda/bin:`$PATH; cd ~/nl2cuda && CUDA_VISIBLE_DEVICES=0 python framework/smoke_test.py && CUDA_VISIBLE_DEVICES=0 python skill/scripts/verify_case.py --case rbf"
  ```
  > 嫌引号烦：把远程命令写进一个 `run.sh` 放远程，本地只 `ssh gpubox "bash ~/run.sh"`。

> **⚠️ 有些 GPU 机默认不暴露 GPU**（`torch.cuda.is_available()` 为 False，须显式 `CUDA_VISIBLE_DEVICES=<卡>` 才可见）。**且 agent 的自动测试机制的子进程未必继承 shell 的 `export`**——把 `CUDA_VISIBLE_DEVICES=<卡> CUDA_ARCHS=<架构>` 直接写进自测命令每条前缀。
> `framework/loader.py` 默认为 sm_75(T4)+sm_80(A100) 编译；其他架构 `export CUDA_ARCHS="架构号"`（A100=80、H100=90）。

rbf `verify` 全 PASS 即环境就绪，进步骤 2。

## 步骤 2：告诉 agent 环境 + 描述算法

直接告诉 agent 两件事（多数 agent 自动读约定文件，无需长提示词）：
```
我的 GPU 环境：<本机有卡，直接跑 / 远程 SSH，用 "ssh gpubox" 一步可登、代码 rsync 到 ~/nl2cuda>。
<然后用自然语言描述你的算法>
例如："把每行向量归一化到均值0方差1再乘 gamma 加 beta，输入 [批,特征] fp32，对输入和 gamma/beta 求梯度。"
```
agent 会先推导数学规格呈你确认（唯一人类确认点），再自建 `cases/<name>/`、自主推导反向。
> **兜底**（agent 没自动读约定文件时）：把 A-1 那段兜底提示给它，但删掉"我本地无 GPU 只写代码不跑测试"——B/C 下 agent 应自己跑自测。

## 步骤 3：agent 自主自测到达标（你只在数学确认点介入）

确认数学规格后，agent **自主循环**：写 kernel → 跑 `verify_case.py`（正确性）→ 通过后 `bench_case.py`（性能）→ 未达标按 `loop.md` 优化 → 重跑，直到达标或诚实报边界。你通常无需手动跑命令。

若想自己核验（真 shell 里直接跑，不加 `!`）：
```bash
python skill/scripts/verify_case.py    --case <你的算法名>   # ① 正确性 allclose，须全 PASS（不过先修正确性、别看性能）
python skill/scripts/check_reference.py --case <你的算法名>   # ② reference 静态预检，期望 REF_CHECK=CLEAN（防弱 baseline 假象）
python skill/scripts/bench_case.py     --case <你的算法名>   # ③ 性能，前反向各自 ≥1.05× torch.compile 才达标
python skill/scripts/profile_case.py   --case <你的算法名>   # ④ 未达标时诊断瓶颈（据结果按 loop.md 选手段）
```

- **达标线**：前向、反向各自 ≥1.05×；擦线（1.05–1.10×）建议连跑 3 次都过。
- **⚠️ 短核假象（务必看）**：若 `bench_case.py` 报 baseline 前/反向 **<1ms**，加速比被固定开销抬高、**不可信**——它会打印"短核假象警告"。**小 reduce/归一化/逐元素算子尤其易中招**：默认小规模常报虚高的"1.5×~1.9× PASS"，放大到计算主导区（baseline≥1ms）后常回落 <1.05。**做法**：`config.py` 让规模支持 env 覆盖，大规模复测（如 `RMS_B=262144 python skill/scripts/bench_case.py --case <名>`）+ 换 2× 规模看稳不稳。
- **本征边界**：纯访存前向 / 厂商库前向手写打不过是正常的——`SKILL.md`"识别本征边界"讲了何时该停、诚实报边界。

---

# 取得 `.cu`（两条路径共用）

达标（或确认边界）后，你的 CUDA kernel 就在：
```
cases/<你的算法名>/kernels/*.cu
```
这已是可用的 `.cu`。若要一份**不依赖 PyTorch、可独立 `nvcc` 编译**的交付版（像 `cases/rbf/delivery/`），照该样例结构整理（可让 agent 做）：

| `cases/rbf/delivery/` 里的文件 | 作用 |
|------|------|
| `rbf_kernels.cu` | 前向+反向 `__global__` kernel + `extern "C"` 裸指针 host 接口（内部管理 device 内存） |
| `rbf_test.cu` | 自测 harness：内置 CPU 参考对拍 GPU 输出，`allclose` 打印 PASS/FAIL（不依赖 torch） |
| `Makefile` | `make test` 一键 nvcc 独立编译 + 跑自测；`make ARCH=sm_75 test` 换架构 |
| `README.md` | 算法说明 + 接口 + 编译运行方式 + 合规声明 |

独立交付版验证（在有 GPU 的环境；Colab 里前面加 `!`）：
```bash
cd cases/<你的算法名>/delivery && make test        # 编译 + CPU 对拍，应打印 PASS
```

**至此你获得了最终的 `.cu`**，可嵌入你自己的 C++/CUDA 项目。

---

## 常见坑

- **`ninja: command not found`**：`pip install ninja`（Colab 默认没装；torch 的 JIT 编译后端必需）。
- **`需要 CUDA GPU`**：当前环境无 GPU 或 `CUDA_VISIBLE_DEVICES` 没指对卡。
- **架构不匹配**：新卡（如 H100 sm_90）需 `export CUDA_ARCHS="90"`。
- **短核假象**：baseline 前/反向 <1ms 却给 1.2×+ 高加速比，多半是固定开销虚高、非真达标。把 `config.py` 规模调大（进 baseline ≥1ms 计算主导区）重测才可信。
- **弱 baseline 假象**：`check_reference.py` 报 WARN 时先核查——reference 若用 Python for 遍历张量维、O(T²) 密集矩阵伪向量化、或按规模切实现分支，会让 baseline 畸形慢、加速比虚高不诚实。reference 必须单一、最干净的向量化写法。
- **红线**：全程 fp32、不用 fast-math、不改 `framework/`；cuBLAS/cuSOLVER 只当积木（GEMM/TRSM/scan）自己拼算法，不得直调与目标算子等价的库成品（如 Cholesky 直调 `potrf`）。

## 一句话流程

**clone 仓库 → 选路径备好 GPU 环境（A=Colab 两阶段 / B=本机有卡 / C=SSH 远程，都跑通 rbf）→ 用自然语言描述算法、agent 推导数学规格请你确认 → agent 自建 `cases/<name>/` 自测到达标（A 路径你半自动搬运 Colab、B/C 路径 agent 自主）→ 从 `kernels/*.cu` 取得 `.cu`（可选整理成 `delivery/` 独立编译版）。**
