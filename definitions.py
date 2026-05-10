import os
from rootutils import find_root

ROOT_DIR = find_root(indicator=".project-root")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CKPT_DIR = os.path.join(ROOT_DIR, "ckpts")
CFG_DIR = os.path.join(ROOT_DIR, "configs")
