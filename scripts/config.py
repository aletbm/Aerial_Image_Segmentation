# config.py
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import ImageColor

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = PROJECT_ROOT / "kaggle" / "input"
DST_PATH = PROJECT_ROOT / "kaggle" / "working"

SIZE_IMG = 256
SIZE_TO_PLOT = 512
BATCH_SIZE = 16

SEG_CLASSES = ["Building", "Land", "Road", "Vegetation", "Water", "Unlabeled", "Unlabeled"]
COLOR_HEX = ["#3C1098", "#8429F6", "#6EC1E4", "#FEDD3A", "#E2A929", "#000000", "#9B9B9B"]
COLOR_RGB = [np.array(ImageColor.getcolor(hex_, "RGB")) for hex_ in COLOR_HEX]

SSAI_COLORS = pd.DataFrame({
    "Classes": SEG_CLASSES,
    "Color_HEX": COLOR_HEX,
    "Color_RGB": COLOR_RGB,
})

AUTOTUNE = tf.data.AUTOTUNE