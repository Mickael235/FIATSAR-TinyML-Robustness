import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

# Confusions critiques du CDC
CRITICAL_PAIRS = {
    ("stop", "go"),
    ("go", "stop"),
    ("left", "right"),
    ("right", "left"),
    ("on", "off"),
    ("off", "on"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--faulty-model", required=True)
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--output", default="f1_eval_fault.csv")
args = parser.parse_args()

baseline = np.load(args.baseline)

Xq = baseline["X_int8"].astype(np.int8)
y = baseline["y"].astype(np.int64)
indices = baseline["indices"].astype(np.int64)
scores_base = baseline["scores_int8"].astype(np.int8)
preds_base = baseline["preds"].astype(np.int64)

if Xq.shape != (500, 1960):
    raise ValueError(f"X_int8 attendu (500,1960), reçu {Xq.shape}")

interpreter = tf.lite.Interpreter(model_path=args.faulty_model)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

if tuple(inp["shape"]) != (1, 1960) or inp["dtype"] != np.int8:
    raise RuntimeError(f"Entrée inattendue : {inp}")

if tuple(out["shape"]) != (1, 12) or out["dtype"] != np.int8:
    raise RuntimeError(f"Sortie inattendue : {out}")

scores_fault = np.empty((500, 12), dtype=np.int8)
preds_fault = np.empty(500, dtype=np.int64)

for i in range(500):
    interpreter.set_tensor(inp["index"], Xq[i:i+1])
    interpreter.invoke()

    s = interpreter.get_tensor(out["index"])[0].astype(np.int8, copy=True)

    scores_fault[i] = s
    preds_fault[i] = int(np.argmax(s))

correct_base = preds_base == y
correct_fault = preds_fault == y

changed_pred = preds_fault != preds_base
changed_scores = np.any(scores_fault != scores_base, axis=1)

base_acc = float(np.mean(correct_base))
fault_acc = float(np.mean(correct_fault))

new_errors = (~correct_fault) & correct_base
recovered = correct_fault & (~correct_base)

critical_count = 0
critical_rows = []

for i in range(500):
    true_name = LABELS[int(y[i])]
    pred_name = LABELS[int(preds_fault[i])]

    is_critical = (
        (true_name, pred_name) in CRITICAL_PAIRS
        or (
            true_name not in ("_silence_", "_unknown_")
            and pred_name == "_silence_"
        )
    )

    if is_critical:
        critical_count += 1
        critical_rows.append(i)

rows = []

for i in range(500):
    diff = (
        scores_fault[i].astype(np.int16)
        - scores_base[i].astype(np.int16)
    )

    true_name = LABELS[int(y[i])]
    base_name = LABELS[int(preds_base[i])]
    fault_name = LABELS[int(preds_fault[i])]

    critical = int(
        (true_name, fault_name) in CRITICAL_PAIRS
        or (
            true_name not in ("_silence_", "_unknown_")
            and fault_name == "_silence_"
        )
    )

    rows.append({
        "sample_order": i,
        "dataset_index": int(indices[i]),
        "true_label_id": int(y[i]),
        "true_label": true_name,
        "baseline_pred_id": int(preds_base[i]),
        "baseline_pred": base_name,
        "faulty_pred_id": int(preds_fault[i]),
        "faulty_pred": fault_name,
        "baseline_correct": int(correct_base[i]),
        "faulty_correct": int(correct_fault[i]),
        "prediction_changed": int(changed_pred[i]),
        "scores_changed": int(changed_scores[i]),
        "new_error": int(new_errors[i]),
        "recovered_error": int(recovered[i]),
        "critical_confusion": critical,
        "max_abs_score_diff": int(np.max(np.abs(diff))),
    })

with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("===== EVALUATION F1 =====")
print("Modele fauté :", args.faulty_model)
print()
print(f"Accuracy baseline : {100*base_acc:.2f}% ({int(correct_base.sum())}/500)")
print(f"Accuracy fautée   : {100*fault_acc:.2f}% ({int(correct_fault.sum())}/500)")
print(f"Delta accuracy    : {100*(fault_acc-base_acc):+.2f} points")
print()
print(f"Predictions changées : {int(changed_pred.sum())}/500")
print(f"Scores changés        : {int(changed_scores.sum())}/500")
print(f"Nouvelles erreurs     : {int(new_errors.sum())}")
print(f"Erreurs corrigées     : {int(recovered.sum())}")
print(f"Confusions critiques  : {critical_count}")
print()
print("CSV :", args.output)

if changed_pred.any():
    print()
    print("Premiers changements de prédiction :")
    for i in np.flatnonzero(changed_pred)[:10]:
        print(
            f"- sample {i:3d} / dataset {indices[i]:4d} | "
            f"vrai={LABELS[int(y[i])]:10s} | "
            f"base={LABELS[int(preds_base[i])]:10s} | "
            f"faute={LABELS[int(preds_fault[i])]:10s}"
        )
