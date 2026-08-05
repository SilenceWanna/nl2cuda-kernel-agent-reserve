import os


_SIZE = int(os.environ.get("II_SIZE", "4096"))
H = int(os.environ.get("II_H", str(_SIZE)))
W = int(os.environ.get("II_W", str(_SIZE)))
DTYPE = "float32"
