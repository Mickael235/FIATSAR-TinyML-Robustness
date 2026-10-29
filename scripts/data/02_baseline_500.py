import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="model_original.tflite")
parser.add_argument("--testset", default="testset_500.npz")
parser.add_argument("--csv", default="baseline_500.csv")
parser.add_argument("--npz", default="baseline_500.npz")
parser.add_argument("--meta", default="baseline_500_metadata.json")
args = parser.parse_args()

model_path = Path(args.model)
test_path = Path(args.testset)

model_bytes = model_path.read_bytes()
model_sha256 = hashlib.sha256(model_bytes).hexdigest()

d = np.load(test_path)
X = d["X"].astype(np.float32)
y = d["y"].astype(np.int64)
indices = d["indices"].astype(np.int64)

if X.shape != (500, 1960):
    raise ValueError(f"Testset attendu (500,1960), reçu {X.shape}")
if y.shape != (500,):
    raise ValueError(f"Labels attendus (500,), reçus {y.shape}")

interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

if tuple(inp["shape"]) != (1, 1960) or inp["dtype"] != np.int8:
    raise RuntimeError(f"Entrée inattendue : {inp}")
if tuple(out["shape"]) != (1, 12) or out["dtype"] != np.int8:
    raise RuntimeError(f"Sortie inattendue : {out}")

in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

if in_scale <= 0:
    raise RuntimeError("Quantification d'entrée invalide")

Xq = np.round(X / in_scale + in_zero)
Xq = np.clip(Xq, -128, 127).astype(np.int8)

scores = np.empty((500, 12), dtype=np.int8)
preds = np.empty(500, dtype=np.int64)

for i in range(500):
    interpreter.set_tensor(inp["index"], Xq[i:i+1])
    interpreter.invoke()
    s = interpreter.get_tensor(out["index"])[0].astype(np.int8, copy=True)
    scores[i] = s
    preds[i] = int(np.argmax(s))

correct = preds == y
accuracy = float(np.mean(correct))

fieldnames = [
    "sample_order", "dataset_index",
    "true_label_id", "true_label",
    "pred_id", "pred_label", "correct",
] + [f"score_{i}" for i in range(12)]

with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    for i in range(500):
        row = {
            "sample_order": i,
            "dataset_index": int(indices[i]),
            "true_label_id": int(y[i]),
            "true_label": LABELS[int(y[i])],
            "pred_id": int(preds[i]),
            "pred_label": LABELS[int(preds[i])],
            "correct": int(correct[i]),
        }
        for j in range(12):
            row[f"score_{j}"] = int(scores[i, j])
        w.writerow(row)

np.savez_compressed(
    args.npz,
    X_float=X,
    X_int8=Xq,
    y=y,
    indices=indices,
    scores_int8=scores,
    preds=preds,
    correct=correct.astype(np.uint8),
)

metadata = {
    "model_file": model_path.name,
    "model_size_bytes": len(model_bytes),
    "model_sha256": model_sha256,
    "testset_file": test_path.name,
    "n_samples": 500,
    "input_shape": inp["shape"].tolist(),
    "input_dtype": str(inp["dtype"]),
    "input_scale": float(in_scale),
    "input_zero_point": int(in_zero),
    "output_shape": out["shape"].tolist(),
    "output_dtype": str(out["dtype"]),
    "output_scale": float(out_scale),
    "output_zero_point": int(out_zero),
    "accuracy": accuracy,
    "accuracy_percent": 100.0 * accuracy,
}

Path(args.meta).write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("===== BASELINE F1 - TINY_CONV ORIGINAL =====")
print("Modèle :", model_path)
print("SHA256 :", model_sha256)
print("Input  :", inp["shape"], inp["dtype"], inp["quantization"])
print("Output :", out["shape"], out["dtype"], out["quantization"])
print(f"Accuracy : {100*accuracy:.2f}% ({int(correct.sum())}/500)")
print("✅", args.csv)
print("✅", args.npz)
print("✅", args.meta)
