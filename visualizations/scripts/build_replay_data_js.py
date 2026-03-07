#!/usr/bin/env python3
"""Bundle all replay JSON files into a single JS file loadable via <script src>.

This avoids fetch() CORS issues when opening HTML from file://.

Usage:
    python visualizations/scripts/build_replay_data_js.py

Output:
    visualizations/output/replay/replay_data.js
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPLAY_DIR = PROJECT_ROOT / "visualizations" / "output" / "replay"
OUTPUT = REPLAY_DIR / "replay_data.js"

DATASETS = ['aime24', 'aime25', 'math500', 'gsm8k', 'humaneval', 'mbpp', 'mt-bench', 'alpaca', 'swe-bench']


def main():
    data = {}
    for ds in DATASETS:
        path = REPLAY_DIR / f"replay_{ds}.json"
        if path.exists():
            with open(path) as f:
                data[ds] = json.load(f)
            print(f"  {ds}: {len(data[ds]['samples'])} samples")
        else:
            print(f"  SKIP {ds}: not found")

    js = "// Auto-generated — do not edit. Run: python visualizations/scripts/build_replay_data_js.py\n"
    js += "window.REPLAY_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n"

    with open(OUTPUT, 'w') as f:
        f.write(js)

    print(f"\nWritten: {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
