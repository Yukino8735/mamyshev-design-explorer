#!/usr/bin/env python3
"""Trusted local Demo model CLI; JSON input/output, prediction only."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mamyshev_ml.demo_inference import predict, MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text())
    result = predict(ROOT / "demo/assets/models", args.model_id, value if isinstance(value, list) else [value])
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"model_id": result["model_id"], "n_predictions": result["n_predictions"], "kind": "Prediction"}))


if __name__ == "__main__":
    main()
