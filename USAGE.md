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
| `skill/DESIGN.md` / `loop.md` / `AUTONOMOUS_LOOP.md` | 架构 / 优化循环 / 自测决策 | 只读 |
| `skill/scripts/verify_case.py` | 正确性验证 CLI | 只读 |
| `skill/scripts/bench_case.py` | 计时 CLI（vs torch.compile） | 只读 |
| `skill/scripts/profile_case.py` | 瓶颈诊断 CLI（优化时用） | 只读 |
| `skill/scripts/check_reference.py` | reference 静态预检（防弱 baseline） | 只读 |
| `cases/rbf/` | 唯一内置完整样例（agent 复制它作结构模板；含 `delivery/` 独立编译成品样例） | 参考 |
| `AGENTS.md` / `CLAUDE.md` / `CONVENTIONS.md` | agent 启动约定（不同宿主自动读对应文件） | 只读 |
| `requirements.txt` / `scripts/probe_env.py` / `notebooks/run.ipynb` | 依赖 / 环境探测 / Colab 入口 | 只读 |

> **你的产物只落在新建的 `cases/<你的算法名>/` 目录**，framework 与上述工具都不动。

---

## 步骤 1：准备 GPU 环境

编译运行 CUDA kernel **需要 NVIDIA GPU + nvcc + PyTorch(cuda) + ninja**。二选一：

### 路径 A（推荐，零成本）：Google Colab 免费 T4

1. 打开 [Colab](https://colab.research.google.com/)，`代码执行程序 → 更改运行时类型 → T4 GPU`。
2. 上传并打开本仓库的 `notebooks/run.ipynb`（或新建 notebook 手敲下面命令）。
3. 依次运行它的单元：clone/pull 仓库 → `pip install ninja` → `python scripts/probe_env.py`（确认 GPU/CUDA/torch）→ `python framework/smoke_test.py`（确认 nvcc+ninja 编译链路通）。

### 路径 B：自备 GPU 机（本地或云）

```bash
pip install -r requirements.txt          # torch>=2.4, numpy>=1.26, ninja>=1.11
# 确认 nvcc 在 PATH：nvcc --version
export CUDA_VISIBLE_DEVICES=0             # 指向一张空闲卡
python scripts/probe_env.py              # 确认 GPU / CUDA / PyTorch 版本
python framework/smoke_test.py           # 确认编译链路（nvcc + ninja）可用
```

> `framework/loader.py` 默认为 sm_75(T4) + sm_80(A100) 编译。其他架构用 `export CUDA_ARCHS="XX"`（如 `90` for H100）。

---

## 步骤 2：冒烟验证（确认环境 OK，用内置 rbf 样例）

先跑通内置样例，确认环境无误再动手做自己的算法：

```bash
python skill/scripts/verify_case.py --case rbf     # 前反向 5 种子 allclose，应全 PASS
python skill/scripts/bench_case.py  --case rbf     # 前反向各自 vs torch.compile 的加速比
```

`verify_case.py` 打印 `PASS` 即环境就绪。（`bench_case.py` 的加速比因卡而异，rbf 参考值前 ~1.10×/反 ~1.17×。）

---

## 步骤 3：驱动 agent 实现你的算法（宿主无关）

把下面这段**上手提示**贴给你的 agent（Claude Code / Codex / aider / Cursor / Cline 等皆可），
替换 `<...>` 为你的算法描述：

```
你将使用本仓库的 nl2cuda-kernel skill，为我把一个算法实现为自定义 CUDA 前向+反向 kernel，
以 PyTorch 为金标准验证正确性，并在规范计时下力争超过 torch.compile。

1. 先读 skill/SKILL.md（方法论主体）和 skill/DESIGN.md（架构），严格按其流程和防作弊红线执行。
   framework/ 对你只读，禁止修改。cases/rbf/ 是完整结构范例，照它的 7 文件骨架来。
2. 我的算法：<自然语言描述前向计算 + 各输入张量名/形状/dtype + 输出 + 对哪些输入求梯度 + 标量参数>
   例如："LayerNorm。每行减均值除标准差(含eps)再乘gamma加beta。输入 X[B,D]、gamma[D]、beta[D]，
   fp32。对 X/gamma/beta 求梯度。"
3. 若我给的是模糊自然语言（无精确公式/shape）、或点了个有多种变体的算子名，先按 SKILL.md 步骤 0.5
   推导数学规格 + PyTorch reference 呈给我确认，再动工（这是唯一的人类确认点）。
4. 在 cases/<name>/ 下写：reference.py（PyTorch 金标准，禁止落回 F.*/SDPA 高层算子）、config.py、
   __init__.py（暴露 CASE）、kernels/*.cu（前向+反向，反向公式自主推导）、op.py（autograd.Function 封装为 candidate）。
5. 用只读 CLI 自测（见步骤 4），未达标按 skill/loop.md 迭代，只改 cases/<name>/。
6. 红线：fp32 全精度、不用 fast-math、不降精度换速度、不改 framework/、cuBLAS/cuSOLVER 只能作
   辅助原语(GEMM/TRSM 等积木)不得直调与目标算子等价的库成品(如分解直调 potrf)。
```

agent 会产出 `cases/<你的算法名>/`（reference.py / config.py / __init__.py / op.py / kernels/*.cu）。

---

## 步骤 4：自测到达标

> **⚠️ 重要（交付物一般性说明）**：`SKILL.md` / 约定文件里提到的 `run_on_a100.sh` 是**本项目作者
> 专用的"远程 A100 自测"封装**（内含作者私有的 SSH 连接），**不包含在通用交付物里、你也用不上**。
> 通用环境下，你**直接在自己的 GPU 上跑下面两个 CLI 自测**即可，效果等价：

```bash
# ① 正确性（必须先过；不过就是数学/kernel 写错，先修这个，别看性能）
python skill/scripts/verify_case.py --case <你的算法名>

# ② reference 静态预检（防"弱 baseline 假象"——reference 写太慢会让加速比虚高）
python skill/scripts/check_reference.py --case <你的算法名>     # 期望 REF_CHECK=CLEAN

# ③ 性能（正确性过了再看；前反向各自 ≥1.05× torch.compile 才算达标）
python skill/scripts/bench_case.py --case <你的算法名>

# ④ 没达标时诊断瓶颈（据结果按 skill/loop.md 选优化手段）
python skill/scripts/profile_case.py --case <你的算法名>
```

- **正确性优先**：`verify_case.py` 不 PASS 时，只修正确性，不看性能。
- **达标线**：前向、反向各自 ≥1.05× `torch.compile`。擦线（1.05–1.10×）建议连跑 3 次都过再算数。
- **未达标**：让 agent 读 `skill/loop.md`，据 `profile_case.py` 的瓶颈诊断选手段（tiling / float4 / warp 规约 / 算子融合 / 前向缓存复用等）迭代，每轮只改 `cases/<name>/`、改完重新 verify（必须仍全 PASS）再 bench。
- **本征边界**：有些算法（纯访存前向、cuSOLVER/cuFFT 级厂商库前向）手写打不过是正常的——`SKILL.md` 的"识别本征边界"章节讲了何时该停、诚实报边界，别空转。

---

## 步骤 5：产出并整理 `.cu` 交付

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

**clone 仓库 → 备好 GPU 环境（Colab 或自备）→ 冒烟跑通 rbf → 把上手提示+算法描述贴给 agent → agent 建 `cases/<name>/` → `verify_case.py`/`bench_case.py` 自测到达标 → 从 `kernels/*.cu` 取得 `.cu`（可选整理成 `delivery/` 独立编译版）。**
