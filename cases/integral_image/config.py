"""integral image（2D 前缀和）case 的算法特定配置（shape / 参数）。

通用计时/容差协议在 framework/protocol.py，跨 case 统一。
形状可切换：默认 H=W=2048（verify 用；2D 前缀和累加 H·W 个 randn，float32 累加误差随规模涨——
4096 时 dX_err ~1.3e-2 会擦破 atol=1e-2，故 verify 默认用 2048（dX_err ~7e-3 稳过，纯浮点累加非逻辑错）。
bench 用 IMG_SIZE=4096（计算主导区，见 bench.env）。env IMG_SIZE 同时设 H/W，IMG_H/IMG_W 可分别覆盖。
"""

import os

_SIZE = int(os.environ.get("IMG_SIZE", "2048"))

H = int(os.environ.get("IMG_H", str(_SIZE)))
W = int(os.environ.get("IMG_W", str(_SIZE)))
DTYPE = "float32"      # 验收精度（禁止降精度换速度）
