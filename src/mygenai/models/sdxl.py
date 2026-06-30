from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline

from .base import BaseModel
from mygenai.utils import get_device, get_dtype


class SDXLModel(BaseModel):
    """Wrapper for Stable Diffusion XL."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        dtype: str = "float32",
    ):
        super().__init__(model_name)
        self.dtype = get_dtype(dtype)
        self.device = get_device()
        self.model: Any = None

    def load(self) -> None:
        """Load SDXL pipeline."""

        self.model = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        self.model.to(self.device)

    def generate(self, prompt: str, num_inference_steps: int = 30):
        """Generate an image from text."""

        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        output = self.model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
        )

        return output.images[0]