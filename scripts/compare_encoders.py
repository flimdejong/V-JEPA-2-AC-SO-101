"""
Compare encoders on the camera. Higher ratio = arm movement stands out more from camera noise.
"""

import time
import numpy as np
import torch
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

MODELS = ["vjepa2_ac_droid", "dino_wm_droid", "jepa_wm_droid"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def grab(camera, n=16):
    frames = []
    for _ in range(n):
        frames.append(camera.read().copy())
        time.sleep(0.05)
    return frames


def look(model, frames):
    """Encode 8 frames and average the latents"""
    with torch.no_grad():
        z = [model.encode(torch.from_numpy(f).permute(2, 0, 1)[None, None]).cpu() for f in frames]
    return torch.stack(z).mean(0)


# Take pictures of two scenes
camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30))
camera.connect()
input("Scene A: arm still, hands out of view. Press Enter...")
a = grab(camera)
input("Scene B: move the arm about 5 cm, hands out of view. Press Enter...")
b = grab(camera)
camera.disconnect()

# Score each model on the same pictures
for name in MODELS:
    model, _ = torch.hub.load("facebookresearch/jepa-wms", name)
    model.to(DEVICE).eval()

    a1, a2, b1 = look(model, a[:8]), look(model, a[8:]), look(model, b[:8])
    floor = (a1 - a2).abs().mean().item()    # same scene twice  -> noise
    signal = (a1 - b1).abs().mean().item()   # scene A vs B      -> arm moved

    print(f"\n{name}: floor {floor:.4f}  signal {signal:.4f}  ratio {signal / floor:.2f}\n")
    del model
    torch.cuda.empty_cache()