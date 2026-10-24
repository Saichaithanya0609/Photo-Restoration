from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def colorize_grayscale(image_bgr: np.ndarray, proto_path: str, model_path: str, points_path: str) -> np.ndarray:
    """Optional OpenCV DNN colourization using user-supplied model files."""
    paths = [Path(proto_path).expanduser(), Path(model_path).expanduser(), Path(points_path).expanduser()]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Colorization requires the Caffe proto, model, and pts_in_hull.npy files")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("colorization expects a BGR colour image")
    net = cv2.dnn.readNetFromCaffe(str(paths[0]), str(paths[1]))
    points = np.load(str(paths[2]))
    class8 = points.transpose().reshape(2, 313, 1, 1).astype(np.float32)
    conv8 = np.float32(np.c_[np.full(313, -2.606), np.zeros(313)])
    net.getLayer(net.getLayerId("class8_ab")).blobs = [class8]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [conv8]
    scaled = image_bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0] * 100.0
    blob = cv2.dnn.blobFromImage(lightness, 1.0, (224, 224), (50,), swapRB=False, crop=False)
    net.setInput(blob)
    ab = net.forward()[0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    output_lab = np.concatenate((lightness[:, :, None], ab), axis=2)
    return np.clip(cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR) * 255.0, 0, 255).astype(np.uint8)
