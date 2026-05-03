"""Download and preprocess the Fama–French 48 industry portfolios for WORK4 §3.

Source: Ken French's data library (Dartmouth Tuck), value-weighted daily
returns. We use VW (not EW) because it's the more standard finance benchmark
and has slightly less microstructure noise.

Pipeline:
    1.  Download the ZIP from Ken French's URL.
    2.  Unzip → CSV with multi-section header.
    3.  Locate the "Average Value Weighted Returns -- Daily" block.
    4.  Drop rows containing the -99.99 missing-data sentinel.
    5.  Convert from percent (e.g. 0.42 = 0.42%) to decimal returns.
    6.  Demean each column (so the multivariate-normal-zero-mean
        assumption holds for our estimators).
    7.  Optionally rescale to unit variance (default: keep raw scale; see
        --standardize).
    8.  Save Y.npy of shape (T, 48), plus metadata.json + a placeholder
        omega_true.npy (= I_48) so the existing data-loading code in
        run_inference_single.py works unchanged.  evaluate_single.py
        reads the ``real_data`` sentinel in metadata.json to skip
        ground-truth metrics for real data.

Usage:
    python scripts/preprocess_real_data.py
    python scripts/preprocess_real_data.py --standardize
    python scripts/preprocess_real_data.py --start-date 2010-01-01 --end-date 2019-12-31
    python scripts/preprocess_real_data.py --offline data/raw/F-F_48_Industry_Portfolios_daily.CSV
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "48_Industry_Portfolios_daily_CSV.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "real" / "ff48"
MISSING_SENTINEL = -99.99


# ======================================================================
# Download + parse
# ======================================================================

def _download_zip(url: str, timeout: int = 30) -> bytes:
    """Fetch the ZIP archive into memory."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (research)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_csv(zip_bytes: bytes) -> str:
    """Pull the single CSV out of the ZIP and return its text content."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV inside the ZIP; got {zf.namelist()}")
        if len(names) > 1:
            print(f"[ff48] warning: ZIP contains {len(names)} CSVs; using {names[0]}")
        with zf.open(names[0]) as f:
            return f.read().decode("latin-1")


def _parse_vw_daily_block(csv_text: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Extract the value-weighted daily returns block.

    The Ken French CSVs have a layout like:

        Average Value Weighted Returns -- Daily
        ,Agric,Food ,Soda ,...
        19260701,0.55,0.34,...
        ...
        Average Equal Weighted Returns -- Daily
        ...

    We locate the VW block, parse its body, stop when we hit the next
    section header.

    Returns
    -------
    data : (T, 48) float64 array of daily returns in percent (NOT yet decimal).
    industry_names : 48 column labels.
    date_strings : T date strings (YYYYMMDD).
    """
    lines = csv_text.splitlines()
    n_lines = len(lines)

    # Find the VW block start.
    vw_start = None
    for i, line in enumerate(lines):
        if "Average Value Weighted Returns -- Daily" in line:
            vw_start = i
            break
    if vw_start is None:
        raise RuntimeError("could not find 'Average Value Weighted Returns -- Daily' header")

    # The next non-empty line after the header is the column header row.
    cursor = vw_start + 1
    while cursor < n_lines and not lines[cursor].strip():
        cursor += 1
    if cursor >= n_lines:
        raise RuntimeError("CSV ended before VW header row")

    header_line = lines[cursor]
    cursor += 1

    # Column 0 is the date. Remaining columns are industry names.
    industry_names = [c.strip() for c in header_line.split(",")[1:]]
    if len(industry_names) != 48:
        raise RuntimeError(
            f"expected 48 industry columns, got {len(industry_names)}: {industry_names[:5]}..."
        )

    # Parse data rows until we hit another section header (a non-numeric
    # first-cell line) or EOF.
    dates: list[str] = []
    rows: list[list[float]] = []
    while cursor < n_lines:
        line = lines[cursor].strip()
        cursor += 1
        if not line:
            # Empty line: could be end of section.  Peek ahead.
            continue
        first_cell = line.split(",")[0].strip()
        if not first_cell.isdigit():
            # Likely the next section's title.
            break
        cells = line.split(",")
        if len(cells) != 49:
            # Skip malformed rows (rare but possible at the boundary).
            continue
        try:
            row = [float(x) for x in cells[1:]]
        except ValueError:
            continue
        dates.append(first_cell)
        rows.append(row)

    if not rows:
        raise RuntimeError("VW daily block produced 0 data rows")

    return np.asarray(rows, dtype=np.float64), industry_names, dates


# ======================================================================
# Cleaning
# ======================================================================

def _drop_missing_rows(
    data: np.ndarray, dates: list[str]
) -> tuple[np.ndarray, list[str], int]:
    """Remove rows containing the -99.99 sentinel."""
    bad_mask = np.any(np.isclose(data, MISSING_SENTINEL), axis=1)
    n_dropped = int(bad_mask.sum())
    keep = ~bad_mask
    return data[keep], [d for d, k in zip(dates, keep) if k], n_dropped


def _filter_date_range(
    data: np.ndarray, dates: list[str], start: str | None, end: str | None
) -> tuple[np.ndarray, list[str]]:
    """Keep rows with YYYYMMDD in [start, end] (inclusive)."""
    if start is None and end is None:
        return data, dates
    s = start.replace("-", "") if start else None
    e = end.replace("-", "") if end else None
    keep = np.array(
        [(s is None or d >= s) and (e is None or d <= e) for d in dates],
        dtype=bool,
    )
    return data[keep], [d for d, k in zip(dates, keep) if k]


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--url", type=str, default=DEFAULT_URL,
        help="Source URL for the Fama–French 48 daily returns ZIP.",
    )
    p.add_argument(
        "--offline", type=Path, default=None,
        help="Skip download and read this local CSV instead.  Useful if the cluster "
             "has no outbound HTTPS or the URL is rate-limited.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Where to write Y.npy + metadata.json + omega_true.npy.",
    )
    p.add_argument(
        "--start-date", type=str, default=None,
        help="Filter rows to dates >= this (YYYY-MM-DD).  Default: keep all.",
    )
    p.add_argument(
        "--end-date", type=str, default=None,
        help="Filter rows to dates <= this (YYYY-MM-DD).  Default: keep all.",
    )
    p.add_argument(
        "--standardize", action="store_true",
        help="Rescale each column to unit variance after demeaning.  Default: false "
             "(keep raw return scale; the estimators handle scale internally).",
    )
    args = p.parse_args()

    # 1. Load CSV (online or offline)
    if args.offline is not None:
        print(f"[ff48] reading offline CSV: {args.offline}")
        csv_text = args.offline.read_text(encoding="latin-1")
    else:
        print(f"[ff48] downloading: {args.url}")
        zip_bytes = _download_zip(args.url)
        print(f"[ff48] downloaded {len(zip_bytes) / 1e6:.1f} MB")
        csv_text = _extract_csv(zip_bytes)

    # 2. Parse VW daily block
    raw, industry_names, dates = _parse_vw_daily_block(csv_text)
    print(f"[ff48] parsed VW daily block: {raw.shape[0]} rows × "
          f"{raw.shape[1]} columns ({dates[0]} – {dates[-1]})")

    # 3. Drop missing-data rows
    cleaned, dates_clean, n_dropped = _drop_missing_rows(raw, dates)
    print(f"[ff48] dropped {n_dropped} rows containing -99.99 sentinel")

    # 4. Date filter (optional)
    cleaned, dates_clean = _filter_date_range(
        cleaned, dates_clean, args.start_date, args.end_date,
    )
    print(f"[ff48] after date filter: {cleaned.shape[0]} rows "
          f"({dates_clean[0]} – {dates_clean[-1]})")

    # 5. Convert from percent to decimal returns
    Y = cleaned / 100.0

    # 6. Demean
    column_mean = Y.mean(axis=0)
    Y = Y - column_mean[None, :]

    # 7. Optional standardisation
    column_std = Y.std(axis=0, ddof=1)
    if args.standardize:
        Y = Y / column_std[None, :]
        print("[ff48] standardised: unit variance per column")

    # 8. Sanity checks
    assert not np.isnan(Y).any(), "NaNs after preprocessing"
    assert not np.isinf(Y).any(), "Infs after preprocessing"
    assert Y.shape[1] == 48
    sample_cov = (Y.T @ Y) / max(Y.shape[0] - 1, 1)
    assert np.allclose(sample_cov, sample_cov.T, atol=1e-12), "sample cov not symmetric"

    # 9. Save
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "Y.npy", Y)
    # Placeholder omega_true so existing data-loading code works unchanged.
    # The real_data sentinel below tells evaluate_single.py to skip
    # ground-truth metrics.
    np.save(output_dir / "omega_true.npy", np.eye(48, dtype=np.float64))

    metadata = {
        "real_data": True,
        "source": "Fama-French 48 industry portfolios (value-weighted daily)",
        "url": args.url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "p": 48,
        "T": int(Y.shape[0]),
        "first_date": dates_clean[0],
        "last_date": dates_clean[-1],
        "industries": industry_names,
        "preprocessing": {
            "demean": True,
            "standardize": bool(args.standardize),
            "missing_dropped": int(n_dropped),
            "missing_sentinel": MISSING_SENTINEL,
        },
        "column_mean_pct": [float(x * 100) for x in column_mean.tolist()],
        "column_std_raw": [float(x) for x in column_std.tolist()],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Sanity-check report
    print(f"\n[ff48] saved to {output_dir}")
    print(f"  Y.npy:         shape {Y.shape}, dtype {Y.dtype}")
    print(f"  Per-column mean (post-demean, ×1e6): "
          f"min={Y.mean(axis=0).min()*1e6:.3f}, max={Y.mean(axis=0).max()*1e6:.3f}")
    print(f"  Per-column std: min={Y.std(axis=0).min():.4f}, "
          f"max={Y.std(axis=0).max():.4f}")
    print(f"  Sample-cov diag mean: {np.diag(sample_cov).mean():.6f}")
    print(f"  Date range: {dates_clean[0]} – {dates_clean[-1]} "
          f"({Y.shape[0]} trading days)")
    print(f"  metadata.json: {output_dir / 'metadata.json'}")
    print(f"  omega_true.npy: I_48 (placeholder; real_data sentinel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
