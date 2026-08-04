"""Shape and scalar configuration for dynamic grid evolution."""

import os


_SIZE = int(os.environ.get("DGE_SIZE", "2048"))

H = int(os.environ.get("DGE_H", str(_SIZE)))
W = int(os.environ.get("DGE_W", str(_SIZE)))
K = float(os.environ.get("DGE_K", "1.0"))
DTYPE = "float32"
