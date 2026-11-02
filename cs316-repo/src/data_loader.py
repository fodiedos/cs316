"""Load the dataset with compact dtypes and measure the memory saved."""

import tracemalloc

import numpy as np
import pandas as pd

from .config import DATA_PATH, FEATURES, OPTIMISED_DTYPES, TARGET


def load_default(path=DATA_PATH):
    """Load the CSV with pandas' default dtypes (baseline only)."""
    return pd.read_csv(path)


def load_optimised(path=DATA_PATH):
    """Load the CSV with the compact dtypes from config.py."""
    df = pd.read_csv(path, dtype=OPTIMISED_DTYPES)
    return df


def _peak_memory_of(func, *args):
    """Run func(*args) and return (result, peak bytes) from tracemalloc."""
    tracemalloc.start()
    result = func(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def memory_comparison(path=DATA_PATH):
    """Compare default and optimised loading; return totals and per-column bytes."""
    df_default, peak_default = _peak_memory_of(load_default, path)
    df_opt, peak_opt = _peak_memory_of(load_optimised, path)

    # deep=True counts the real size of strings, not just pointers
    mem_default = df_default.memory_usage(deep=True)
    mem_opt = df_opt.memory_usage(deep=True)

    per_column = pd.DataFrame({
        "default_dtype": df_default.dtypes.astype(str),
        "default_bytes": mem_default.drop("Index"),
        "optimised_dtype": df_opt.dtypes.astype(str),
        "optimised_bytes": mem_opt.drop("Index"),
    })

    max_diff = float(np.max(np.abs(
        df_default[FEATURES].to_numpy() - df_opt[FEATURES].to_numpy(dtype="float64")
    )))

    return {
        "default_total_bytes": int(mem_default.sum()),
        "optimised_total_bytes": int(mem_opt.sum()),
        "reduction_pct": round(100 * (1 - mem_opt.sum() / mem_default.sum()), 1),
        "default_peak_bytes": int(peak_default),
        "optimised_peak_bytes": int(peak_opt),
        "peak_reduction_pct": round(100 * (1 - peak_opt / peak_default), 1),
        "max_abs_value_difference": max_diff,
        "per_column": per_column,
    }


def chunked_crop_means(path=DATA_PATH, chunksize=500):
    """Per-crop feature means computed chunk by chunk with running sums."""
    sums = None
    counts = None

    for chunk in pd.read_csv(path, dtype=OPTIMISED_DTYPES, chunksize=chunksize):
        grouped = chunk.groupby(TARGET, observed=True)[FEATURES]
        chunk_sum = grouped.sum().astype("float64")
        chunk_cnt = grouped.size()
        # fill_value=0 handles crops missing from a chunk
        sums = chunk_sum if sums is None else sums.add(chunk_sum, fill_value=0)
        counts = chunk_cnt if counts is None else counts.add(chunk_cnt, fill_value=0)

    return sums.div(counts, axis=0)


def to_model_arrays(df):
    """Return a float32 feature matrix X and a label array y."""
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].astype(str).to_numpy()
    return X, y
