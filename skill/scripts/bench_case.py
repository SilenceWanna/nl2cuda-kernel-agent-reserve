"""通用计时 CLI：对指定 case 的候选实现 vs torch.compile 前反向计时。

用法：
    python skill/scripts/bench_case.py --case rbf
    python skill/scripts/bench_case.py --case rbf --impl cases.rbf.op:candidate

--case  <name>        : 加载 cases/<name>，取其 CASE 实例
--impl  <module:fn>   : 候选实现 "模块路径:函数名"；默认 cases/<name>/op.py:candidate
"""

import sys
import os
import re
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from framework.bench import compare  # noqa: E402


def load_case(name):
    return importlib.import_module(f"cases.{name}").CASE


def load_impl(spec):
    mod_path, fn_name = spec.split(":")
    return getattr(importlib.import_module(mod_path), fn_name)


def _baseline_ms(report):
    """从报告解析 baseline 前/反向绝对耗时(ms)。用于短核假象检测。
    报告片段： "[baseline(torch.compile)]\n  forward : 0.0652 ms ...\n  backward: 0.1834 ms ..."
    返回 (fwd_ms, bwd_ms)，解析不到则 (None, None)。
    """
    m = re.search(r"\[baseline\(torch\.compile\)\][\s\S]*?forward\s*:\s*([0-9.]+)\s*ms[\s\S]*?backward\s*:\s*([0-9.]+)\s*ms", report)
    if not m:
        return (None, None)
    return (float(m.group(1)), float(m.group(2)))


# baseline 计时低于此值(ms)即判"短核"——固定开销(kernel launch/同步)占比过高，加速比不可信。
# 检测到短核 → 显著警告 + 达标判定存疑(SHORT_KERNEL_SUSPECT)，提示放大规模到计算主导区复测。
SHORT_KERNEL_MS = 1.0


def _parse_verdict(report, passed):
    """从 compare() 的报告文本解析机读摘要。

    passed（来自 framework/bench.compare()，权威）决定 PASS vs BENCH_FAIL；
    报告中的 "存在 CV>5%" 标记 → cv_ok=0 → CV_INVALID（让调用方重测而非信噪声）。
    短核（baseline 前或反向 <1ms）下的 PASS → SHORT_KERNEL_SUSPECT（加速比被固定开销抬高，不算真达标；
    逼调用方放大规模到计算主导区复测）。
    加速比锚在 "x  (需" 上，避免误抓 "... ms" 计时行。
    返回单行字符串，形如 "VERDICT=PASS fwd=1.09x bwd=1.40x cv_ok=1"。
    """
    # 前向/反向加速比行： "  forward : 1.0921x  (需 ≥1.05x) PASS"
    speedups = re.findall(r"(forward|backward)\s*:\s*([0-9.]+)x\s*\(需", report)
    sp = {k: v for k, v in speedups}
    fwd = sp.get("forward", "?")
    bwd = sp.get("backward", "?")
    cv_ok = 0 if "存在 CV>5%" in report else 1

    bf_ms, bb_ms = _baseline_ms(report)
    is_short = bf_ms is not None and (bf_ms < SHORT_KERNEL_MS or bb_ms < SHORT_KERNEL_MS)

    if cv_ok == 0:
        verdict = "CV_INVALID"
    elif passed and is_short:
        verdict = "SHORT_KERNEL_SUSPECT"   # 短核虚高，非真达标——须放大规模复测
    elif passed:
        verdict = "PASS"
    else:
        verdict = "BENCH_FAIL"
    return f"VERDICT={verdict} fwd={fwd}x bwd={bwd}x cv_ok={cv_ok}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case 名，如 rbf")
    ap.add_argument("--impl", default=None, help="候选实现 module:fn；默认 cases.<case>.op:candidate")
    ap.add_argument("--strict", action="store_true",
                    help="达标 exit 0、未达标 exit 1（供自动闭环用退出码判定）")
    ap.add_argument("--emit-verdict", action="store_true",
                    help="报告后额外打印单行机读摘要 VERDICT=...")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("需要 CUDA GPU。本地无 GPU，请在 Colab 运行。")
        return 0

    case = load_case(args.case)
    impl_spec = args.impl or f"cases.{args.case}.op:candidate"
    impl = load_impl(impl_spec)

    print(f"case={args.case}  impl={impl_spec}")
    report, passed = compare(case, impl, device="cuda")
    print(report)

    # 短核假象检测：baseline 前/反向 <1ms 时，加速比被固定开销(launch/同步)抬高、不可信。
    # 通用交付版无 auto-scale 自动放大能力，故显著警告 + 提示放大规模复测（诚实性关键，E2E #11）。
    bf_ms, bb_ms = _baseline_ms(report)
    is_short = bf_ms is not None and (bf_ms < SHORT_KERNEL_MS or bb_ms < SHORT_KERNEL_MS)
    if is_short:
        size_hint = ""
        # 提示 config 里可放大的规模 env（如 RMS_B/LN_B/RBF_SIZE），从 config 模块的 os.environ.get 键名探测
        try:
            cfg = importlib.import_module(f"cases.{args.case}.config")
            src = ""
            cfg_path = getattr(cfg, "__file__", None)
            if cfg_path and os.path.exists(cfg_path):
                with open(cfg_path, encoding="utf-8") as fh:
                    src = fh.read()
            envs = re.findall(r"os\.environ\.get\(\s*[\"']([A-Z_]+)[\"']", src)
            if envs:
                size_hint = f"（放大规模用环境变量：{'、'.join(dict.fromkeys(envs))}，如 {envs[0]}=262144）"
        except Exception:
            pass
        print("\n" + "=" * 70)
        print("⚠️  短核假象警告：baseline 前向 {:.4f}ms / 反向 {:.4f}ms，存在 <1ms 的短核".format(
            bf_ms if bf_ms is not None else -1, bb_ms if bb_ms is not None else -1))
        print("   固定开销(kernel launch/同步)占比过高 → 加速比被抬高、不可信、不算达标。")
        print(f"   请放大规模到计算主导区(baseline 前反向均 ≥1ms)再复测{size_hint}。")
        print("   小 reduce/归一化/逐元素类算子尤其易撞短核假象与带宽墙——放大后加速比常大幅回落。")
        print("=" * 70)

    if args.emit_verdict:
        print(_parse_verdict(report, passed))

    # 默认行为不变（无 flag 恒 exit 0，仅报告）；--strict 才用退出码表达达标与否。
    # --strict 下：短核 PASS 视为未达标（exit 1）——逼用户放大规模优化 kernel 本体，别靠短核凑 PASS。
    if args.strict:
        if passed and is_short:
            return 1
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
