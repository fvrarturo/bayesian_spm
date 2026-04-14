"""IO helpers for posterior sample arrays.

Posterior sample arrays (``omega_samples``, ``kappa_samples`` etc.) are
written as compressed ``.npz`` (single array, key ``"arr"``) to keep the
cluster's disk footprint manageable.  Earlier runs wrote ``.npy``; the
loaders here try ``.npz`` first and fall back to ``.npy`` so un-migrated
directories keep working.

Small files (``omega_hat.npy``, ``sigma_hat.npy``, ``offdiag_magnitudes.npy``)
stay as plain ``.npy`` — they're referenced by many scripts and the
compression win is negligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]

_NPZ_KEY = "arr"


def load_samples(dir_path: PathLike, name: str) -> np.ndarray:
    """Load a sample array from ``dir_path``.

    Tries ``<dir>/<name>.npz`` first, then ``<dir>/<name>.npy``.

    Raises ``FileNotFoundError`` if neither exists.
    """
    dir_path = Path(dir_path)
    npz = dir_path / f"{name}.npz"
    if npz.exists():
        with np.load(npz) as z:
            key = _NPZ_KEY if _NPZ_KEY in z.files else z.files[0]
            return z[key]
    npy = dir_path / f"{name}.npy"
    if npy.exists():
        return np.load(npy)
    raise FileNotFoundError(f"Neither {npz} nor {npy} exists")


def samples_exist(dir_path: PathLike, name: str) -> bool:
    """Return True if either ``<name>.npz`` or ``<name>.npy`` is present."""
    dir_path = Path(dir_path)
    return (dir_path / f"{name}.npz").exists() or (dir_path / f"{name}.npy").exists()


def save_samples_compressed(
    path: PathLike,
    arr: np.ndarray,
    dtype: "np.dtype | None" = None,
) -> None:
    """Write ``arr`` to ``path`` (should end in ``.npz``) using zlib compression.

    If ``dtype`` is given, the array is cast first — intended for downcasting
    stochastic float64 MCMC draws to float32 (precision loss is well below
    Monte Carlo noise).
    """
    path = Path(path)
    if dtype is not None:
        arr = np.asarray(arr).astype(dtype, copy=False)
    else:
        arr = np.asarray(arr)
    np.savez_compressed(path, **{_NPZ_KEY: arr})
