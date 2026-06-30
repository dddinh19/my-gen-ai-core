from typing import Any


class TextToImagePipeline:
    """Text-to-image generation pipeline."""

    def __init__(self, model: Any):
        self.model = model

    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 30,
    ):
        """Generate an image from a text prompt."""

        if self.model.model is None:
            self.model.load()

        return self.model.generate(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
        )