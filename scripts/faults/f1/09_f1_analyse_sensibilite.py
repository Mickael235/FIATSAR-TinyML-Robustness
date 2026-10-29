import argparse
import csv
from pathlib import Path

import numpy as np

def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def to_int(v):
    return int(float(v))

def to_float(v):
    return float(v)

def normalize_screen(rows, layer):
    out = []
    for r in rows:
        out.append({
            "layer": layer,
            "weight_index": to_int(r["weight_index"]),
            "bit": to_int(r["bit"]),
            "delta_accuracy_points": to_float(r["delta_accuracy_points"]),
            "predictions_changed": to_int(r["predictions_changed"]),
            "new_errors": to_int(r["new_errors"]),
            "new_critical_confusions": to_int(r["new_critical_confusions"]),
        })
    return out

def normalize_valid(rows, layer):
    out = []
    for r in rows:
        item = {
            "layer": layer,
            "weight_index": to_int(r["weight_index"]),
            "bit": to_int(r["bit"]),
            "baseline_accuracy_percent": to_float(r["baseline_accuracy_percent"]),
            "fault_accuracy_percent": to_float(r["fault_accuracy_percent"]),
            "delta_accuracy_points": to_float(r["delta_accuracy_points"]),
            "predictions_changed": to_int(r["predictions_changed"]),
            "new_errors": to_int(r["new_errors"]),
            "recovered_errors": to_int(r["recovered_errors"]),
            "new_critical_confusions": to_int(r["new_critical_confusions"]),
        }
        if "output_class" in r and r["output_class"]:
            item["output_class"] = r["output_class"]
        else:
            item["output_class"] = ""
        out.append(item)
    return out

parser = argparse.ArgumentParser()
parser.add_argument("--conv-screen", default="f1_conv_screen.csv")
parser.add_argument("--conv-valid", default="f1_conv_top50_500.csv")
parser.add_argument("--fc-screen", default="f1_fc_screen.csv")
parser.add_argument("--fc-valid", default="f1_fc_top50_500.csv")
args = parser.parse_args()

for p in [args.conv_screen, args.conv_valid, args.fc_screen, args.fc_valid]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Fichier manquant : {p}")

conv_screen = normalize_screen(read_csv(args.conv_screen), "Conv2D")
fc_screen = normalize_screen(read_csv(args.fc_screen), "FullyConnected")
conv_valid = normalize_valid(read_csv(args.conv_valid), "Conv2D")
fc_valid = normalize_valid(read_csv(args.fc_valid), "FullyConnected")

# ------------------------------------------------------------
# 1) Sensibilité par couche et bit à partir du SCREEN
# ------------------------------------------------------------
summary_rows = []

for layer, rows in [("Conv2D", conv_screen), ("FullyConnected", fc_screen)]:
    for bit in range(7, -1, -1):
        subset = [r for r in rows if r["bit"] == bit]
        if not subset:
            continue

        deltas = np.array([r["delta_accuracy_points"] for r in subset], dtype=float)
        changed = np.array([r["predictions_changed"] for r in subset], dtype=int)
        newerr = np.array([r["new_errors"] for r in subset], dtype=int)
        crit = np.array([r["new_critical_confusions"] for r in subset], dtype=int)

        summary_rows.append({
            "layer": layer,
            "bit": bit,
            "n_faults_screened": len(subset),
            "mean_delta_accuracy_points": float(np.mean(deltas)),
            "median_delta_accuracy_points": float(np.median(deltas)),
            "worst_delta_accuracy_points": float(np.min(deltas)),
            "best_delta_accuracy_points": float(np.max(deltas)),
            "faults_changing_prediction": int(np.sum(changed > 0)),
            "percent_faults_changing_prediction": 100.0 * float(np.mean(changed > 0)),
            "faults_creating_new_error": int(np.sum(newerr > 0)),
            "percent_faults_creating_new_error": 100.0 * float(np.mean(newerr > 0)),
            "faults_creating_new_critical": int(np.sum(crit > 0)),
            "percent_faults_creating_new_critical": 100.0 * float(np.mean(crit > 0)),
            "max_predictions_changed": int(np.max(changed)),
            "max_new_errors": int(np.max(newerr)),
            "max_new_critical": int(np.max(crit)),
        })

with open("f1_sensibilite_par_bit.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    w.writeheader()
    w.writerows(summary_rows)

# ------------------------------------------------------------
# 2) Résumé par couche à partir des fautes validées sur 500
# ------------------------------------------------------------
layer_rows = []

for layer, rows in [("Conv2D", conv_valid), ("FullyConnected", fc_valid)]:
    deltas = np.array([r["delta_accuracy_points"] for r in rows], dtype=float)
    changed = np.array([r["predictions_changed"] for r in rows], dtype=int)
    newerr = np.array([r["new_errors"] for r in rows], dtype=int)
    crit = np.array([r["new_critical_confusions"] for r in rows], dtype=int)

    worst = min(rows, key=lambda r: (
        r["delta_accuracy_points"],
        -r["new_critical_confusions"],
        -r["predictions_changed"]
    ))

    layer_rows.append({
        "layer": layer,
        "n_validated_faults": len(rows),
        "mean_delta_accuracy_points": float(np.mean(deltas)),
        "worst_delta_accuracy_points": float(np.min(deltas)),
        "mean_predictions_changed": float(np.mean(changed)),
        "mean_new_errors": float(np.mean(newerr)),
        "faults_with_new_critical": int(np.sum(crit > 0)),
        "worst_weight_index": worst["weight_index"],
        "worst_bit": worst["bit"],
        "worst_output_class": worst.get("output_class", ""),
        "worst_fault_accuracy_percent": worst["fault_accuracy_percent"],
        "worst_predictions_changed": worst["predictions_changed"],
        "worst_new_errors": worst["new_errors"],
        "worst_new_critical": worst["new_critical_confusions"],
    })

with open("f1_resume_couches.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=layer_rows[0].keys())
    w.writeheader()
    w.writerows(layer_rows)

# ------------------------------------------------------------
# 3) Top 10 des candidats VALIDÉS sur 500
# ------------------------------------------------------------
all_valid = conv_valid + fc_valid
ranked = sorted(
    all_valid,
    key=lambda r: (
        r["delta_accuracy_points"],
        -r["new_critical_confusions"],
        -r["predictions_changed"],
        -r["new_errors"],
    )
)

top10 = ranked[:10]

with open("f1_top10_candidats_valides_500.csv", "w", newline="", encoding="utf-8-sig") as f:
    fields = [
        "rank", "layer", "weight_index", "bit", "output_class",
        "baseline_accuracy_percent", "fault_accuracy_percent",
        "delta_accuracy_points", "predictions_changed",
        "new_errors", "recovered_errors", "new_critical_confusions"
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    for rank, r in enumerate(top10, start=1):
        w.writerow({
            "rank": rank,
            **{k: r.get(k, "") for k in fields if k != "rank"}
        })

print("===== RESUME F1 =====")
print()
print("Couches validées sur 500 :")
for r in layer_rows:
    print(
        f"- {r['layer']:14s} | "
        f"pire Δacc={r['worst_delta_accuracy_points']:+.2f} pts | "
        f"faute={r['worst_weight_index']}/bit{r['worst_bit']} | "
        f"predΔ={r['worst_predictions_changed']} | "
        f"newErr={r['worst_new_errors']} | "
        f"newCrit={r['worst_new_critical']}"
    )

print()
print("===== TOP 10 CANDIDATS VALIDÉS SUR 500 =====")
for i, r in enumerate(top10, start=1):
    extra = f" class={r['output_class']}" if r.get("output_class") else ""
    print(
        f"{i:2d}. {r['layer']:14s} "
        f"w={r['weight_index']:5d} bit={r['bit']}{extra} | "
        f"Δacc={r['delta_accuracy_points']:+.2f} pts | "
        f"predΔ={r['predictions_changed']:3d} | "
        f"newErr={r['new_errors']:3d} | "
        f"newCrit={r['new_critical_confusions']:2d}"
    )

print()
print("===== BIT LE PLUS SENSIBLE PAR COUCHE (SCREEN) =====")
for layer in ["Conv2D", "FullyConnected"]:
    rows = [r for r in summary_rows if r["layer"] == layer]
    # critère principal = pire delta observé, puis taux de fautes changeant une prédiction
    best = min(
        rows,
        key=lambda r: (
            r["worst_delta_accuracy_points"],
            -r["percent_faults_changing_prediction"]
        )
    )
    print(
        f"- {layer:14s} : bit {best['bit']} | "
        f"pire Δacc={best['worst_delta_accuracy_points']:+.2f} pts | "
        f"{best['percent_faults_changing_prediction']:.1f}% "
        f"des fautes changent au moins une prédiction"
    )

print()
print("✅ f1_sensibilite_par_bit.csv")
print("✅ f1_resume_couches.csv")
print("✅ f1_top10_candidats_valides_500.csv")
