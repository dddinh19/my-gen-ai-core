import torch

def get_dtype(dtype: str = "float32") -> torch.dtype:
    """Return a torch dtype from its string name."""

    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype not in dtypes:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return dtypes[dtype]