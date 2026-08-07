"""Shape and input-distribution settings for warp_sparse_contrast."""

import os


GROUPS = int(os.environ.get("WSC_GROUPS", "4096"))
WARP_SIZE = 32
T_DENSITY = 0.50
EDGE_DENSITY = 0.25
DTYPE = "float32"
