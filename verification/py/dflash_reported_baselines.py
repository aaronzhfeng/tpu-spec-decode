"""Extracted reported DFlash reference values from external/dflash assets.

Sources:
- external/dflash/assets/dflash_results.png
- external/dflash/assets/speedup.png
- external/dflash/README.md (hardware note: NVIDIA B200 GPUs)
"""

from __future__ import annotations

# category -> model_key -> temperature -> dataset -> {"speedup": float, "tau": float}
TABLE_BASELINES: dict[str, dict[str, dict[int, dict[str, dict[str, float]]]]] = {
    "math": {
        "Qwen3-8B-speculator.eagle3": {
            0: {
                "GSM8K": {"speedup": 2.13, "tau": 2.89},
                "Math500": {"speedup": 2.18, "tau": 2.94},
                "AIME24": {"speedup": 2.25, "tau": 3.04},
                "AIME25": {"speedup": 2.18, "tau": 2.93},
                "Average": {"speedup": 2.19, "tau": 2.95},
            },
            1: {
                "GSM8K": {"speedup": 2.07, "tau": 2.79},
                "Math500": {"speedup": 2.03, "tau": 2.75},
                "AIME24": {"speedup": 1.88, "tau": 2.54},
                "AIME25": {"speedup": 1.81, "tau": 2.44},
                "Average": {"speedup": 1.95, "tau": 2.63},
            },
        },
        "Qwen3-4B-DFlash-b16": {
            0: {
                "GSM8K": {"speedup": 5.17, "tau": 6.50},
                "Math500": {"speedup": 6.19, "tau": 7.84},
                "AIME24": {"speedup": 6.00, "tau": 7.47},
                "AIME25": {"speedup": 5.79, "tau": 7.28},
                "Average": {"speedup": 5.79, "tau": 7.27},
            },
            1: {
                "GSM8K": {"speedup": 4.73, "tau": 5.98},
                "Math500": {"speedup": 5.14, "tau": 6.67},
                "AIME24": {"speedup": 3.84, "tau": 4.97},
                "AIME25": {"speedup": 3.89, "tau": 5.01},
                "Average": {"speedup": 4.40, "tau": 5.66},
            },
        },
        "Qwen3-8B-DFlash-b16": {
            0: {
                "GSM8K": {"speedup": 5.20, "tau": 6.55},
                "Math500": {"speedup": 6.17, "tau": 7.87},
                "AIME24": {"speedup": 5.91, "tau": 7.48},
                "AIME25": {"speedup": 5.85, "tau": 7.31},
                "Average": {"speedup": 5.78, "tau": 7.30},
            },
            1: {
                "GSM8K": {"speedup": 4.78, "tau": 6.04},
                "Math500": {"speedup": 5.02, "tau": 6.57},
                "AIME24": {"speedup": 3.87, "tau": 5.06},
                "AIME25": {"speedup": 3.84, "tau": 5.03},
                "Average": {"speedup": 4.38, "tau": 5.68},
            },
        },
    },
    "code": {
        "Qwen3-8B-speculator.eagle3": {
            0: {
                "Humaneval": {"speedup": 2.48, "tau": 3.36},
                "MBPP": {"speedup": 2.27, "tau": 3.08},
                "LiveCodeBench": {"speedup": 2.24, "tau": 3.16},
                "SWE-Bench": {"speedup": 1.90, "tau": 2.55},
                "Average": {"speedup": 2.22, "tau": 3.04},
            },
            1: {
                "Humaneval": {"speedup": 2.30, "tau": 3.11},
                "MBPP": {"speedup": 2.15, "tau": 2.92},
                "LiveCodeBench": {"speedup": 2.17, "tau": 3.00},
                "SWE-Bench": {"speedup": 1.66, "tau": 2.21},
                "Average": {"speedup": 2.07, "tau": 2.81},
            },
        },
        "Qwen3-4B-DFlash-b16": {
            0: {
                "Humaneval": {"speedup": 5.26, "tau": 6.63},
                "MBPP": {"speedup": 4.87, "tau": 6.19},
                "LiveCodeBench": {"speedup": 5.41, "tau": 6.97},
                "SWE-Bench": {"speedup": 2.97, "tau": 3.70},
                "Average": {"speedup": 4.63, "tau": 5.87},
            },
            1: {
                "Humaneval": {"speedup": 4.80, "tau": 6.05},
                "MBPP": {"speedup": 4.35, "tau": 5.55},
                "LiveCodeBench": {"speedup": 5.00, "tau": 6.60},
                "SWE-Bench": {"speedup": 2.51, "tau": 3.09},
                "Average": {"speedup": 4.17, "tau": 5.32},
            },
        },
        "Qwen3-8B-DFlash-b16": {
            0: {
                "Humaneval": {"speedup": 5.20, "tau": 6.55},
                "MBPP": {"speedup": 4.75, "tau": 6.00},
                "LiveCodeBench": {"speedup": 5.43, "tau": 7.12},
                "SWE-Bench": {"speedup": 2.92, "tau": 3.69},
                "Average": {"speedup": 4.58, "tau": 5.84},
            },
            1: {
                "Humaneval": {"speedup": 4.35, "tau": 5.40},
                "MBPP": {"speedup": 4.07, "tau": 5.17},
                "LiveCodeBench": {"speedup": 5.15, "tau": 6.79},
                "SWE-Bench": {"speedup": 2.30, "tau": 2.82},
                "Average": {"speedup": 3.97, "tau": 5.05},
            },
        },
    },
    "chat": {
        "Qwen3-8B-speculator.eagle3": {
            0: {
                "MT-Bench": {"speedup": 1.94, "tau": 2.72},
                "Alpaca": {"speedup": 1.88, "tau": 2.68},
                "Average": {"speedup": 1.91, "tau": 2.70},
            },
            1: {
                "MT-Bench": {"speedup": 1.81, "tau": 2.55},
                "Alpaca": {"speedup": 1.79, "tau": 2.56},
                "Average": {"speedup": 1.80, "tau": 2.56},
            },
        },
        "Qwen3-4B-DFlash-b16": {
            0: {
                "MT-Bench": {"speedup": 2.87, "tau": 4.35},
                "Alpaca": {"speedup": 2.23, "tau": 3.10},
                "Average": {"speedup": 2.55, "tau": 3.73},
            },
            1: {
                "MT-Bench": {"speedup": 2.63, "tau": 4.03},
                "Alpaca": {"speedup": 2.16, "tau": 2.99},
                "Average": {"speedup": 2.40, "tau": 3.51},
            },
        },
        "Qwen3-8B-DFlash-b16": {
            0: {
                "MT-Bench": {"speedup": 2.79, "tau": 4.25},
                "Alpaca": {"speedup": 2.27, "tau": 3.16},
                "Average": {"speedup": 2.53, "tau": 3.71},
            },
            1: {
                "MT-Bench": {"speedup": 2.50, "tau": 3.74},
                "Alpaca": {"speedup": 2.11, "tau": 2.88},
                "Average": {"speedup": 2.31, "tau": 3.31},
            },
        },
    },
}

# Extracted from external/dflash/assets/speedup.png.
PLOT_SPEEDUP_BASELINE_QWEN3_8B = {
    "GSM8K": {"autoregressive": 1.00, "eagle3": 2.13, "dflash": 5.10},
    "Math500": {"autoregressive": 1.00, "eagle3": 2.18, "dflash": 6.09},
    "AIME24": {"autoregressive": 1.00, "eagle3": 2.25, "dflash": 5.73},
    "AIME25": {"autoregressive": 1.00, "eagle3": 2.18, "dflash": 5.75},
    "Humaneval": {"autoregressive": 1.00, "eagle3": 2.48, "dflash": 5.18},
    "MBPP": {"autoregressive": 1.00, "eagle3": 2.27, "dflash": 4.67},
    "LiveCodeBench": {"autoregressive": 1.00, "eagle3": 2.24, "dflash": 5.51},
    "SWE-Bench": {"autoregressive": 1.00, "eagle3": 1.90, "dflash": 2.88},
    "MT-Bench": {"autoregressive": 1.00, "eagle3": 1.94, "dflash": 2.76},
    "Alpaca": {"autoregressive": 1.00, "eagle3": 1.88, "dflash": 2.21},
}

