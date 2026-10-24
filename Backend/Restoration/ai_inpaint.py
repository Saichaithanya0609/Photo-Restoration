from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class OptionalLamaInpainting:
    """Small adapter for a user-supplied TorchScript LaMa model.

    The model is never downloaded implicitly. Set ``LAMA_MODEL`` or pass a
    model path to the pipeline. The expected artifact is the TorchScript
    ``big-lama.pt`` model published by the simple-lama-inpainting project.
    """

    def __init__(self, model_path: str, device: str = "auto", max_side: int = 1600) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("AI inpainting requires optional PyTorch. Install the Apple Silicon instructions in README.md.") from exc
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LaMa model not found: {path}")
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.max_side = int(np.clip(max_side, 256, 4096))
        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = np.asarray(image_bgr, dtype=np.uint8)
        original_height, original_width = image.shape[:2]
        scale = min(1.0, self.max_side / max(original_height, original_width))
        if scale < 1.0:
            width = max(32, int(round(original_width * scale)))
            height = max(32, int(round(original_height * scale)))
            working = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            working_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            working, working_mask = image.copy(), mask.copy()

        height, width = working.shape[:2]
        padded_height = int(np.ceil(height / 8.0) * 8)
        padded_width = int(np.ceil(width / 8.0) * 8)
        padded = cv2.copyMakeBorder(working, 0, padded_height - height, 0, padded_width - width, cv2.BORDER_REFLECT101)
        padded_mask = cv2.copyMakeBorder(working_mask, 0, padded_height - height, 0, padded_width - width, cv2.BORDER_CONSTANT, value=0)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_tensor = self.torch.from_numpy((padded_mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        masked_tensor = image_tensor * (1.0 - mask_tensor)
        with self.torch.inference_mode():
            prediction = self.model(masked_tensor, mask_tensor)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        if prediction.ndim == 4:
            prediction = prediction[0]
        output_rgb = prediction.detach().float().cpu().permute(1, 2, 0).numpy()
        output_rgb = np.clip(output_rgb * 255.0, 0, 255).astype(np.uint8)[:height, :width]
        output = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        if scale < 1.0:
            output = cv2.resize(output, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
        return output


def ai_backend_available(model_path: str | None) -> bool:
    return bool(model_path and Path(model_path).expanduser().is_file())
