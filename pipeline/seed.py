"""Global reproducibility seeding for the hybrid QML pneumonia pipeline.

Convention: every experiment in this project uses ``SEED = 6``. All random
sources (Python ``random``, NumPy, PyTorch CPU/CUDA, DataLoader shuffling)
must be seeded with this value so that runs are reproducible end-to-end.

All heavy library imports (``torch``) are performed lazily *inside* the
functions so that this module can be imported even in environments where
PyTorch is not installed (e.g. on the CI/development machine).
"""

import os
import random

SEED = 6


def seed_everything(seed: int = 6) -> None:
    """Seed all random sources for full reproducibility.

    Seeds Python ``random``, sets ``PYTHONHASHSEED``, seeds NumPy, seeds
    PyTorch CPU and all CUDA devices, and forces deterministic cuDNN
    behaviour (``cudnn.deterministic = True``, ``cudnn.benchmark = False``).

    Args:
        seed: Global random seed. Defaults to the project convention ``6``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        # PyTorch is optional at import time (e.g. on the dev machine).
        # When it is absent, skip the torch-specific seeding.
        import warnings

        warnings.warn(
            "PyTorch not installed; skipping torch/cuDNN seeding.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_torch_generator(seed: int = 6) -> "torch.Generator":
    """Return a seeded ``torch.Generator`` for reproducible DataLoader shuffling.

    Args:
        seed: Random seed. Defaults to the project convention ``6``.

    Returns:
        A ``torch.Generator`` seeded with ``seed``, ready to be passed as the
        ``generator`` argument of a ``torch.utils.data.DataLoader``.
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator