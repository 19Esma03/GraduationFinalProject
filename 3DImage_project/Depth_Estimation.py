from transformers import pipeline
import numpy as np
from PIL import Image

_depth_estimator = None

def get_depth_map(image):
    global _depth_estimator

    if _depth_estimator is None:
        print("Loading depth estimation model...")
        _depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    print("Estimating depth map...")

    try:
        depth_map = _depth_estimator(image)
        depth_array = np.array(depth_map["depth"])
        return depth_array
    except Exception as e:
        print(f"Depth Estimation Error: {e}")
        return np.zeros((image.height, image.width))


