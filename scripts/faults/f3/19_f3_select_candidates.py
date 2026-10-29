import argparse
import csv
from pathlib import Path


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def I(r, k):
    return int(float(r[k]))


def F(r, k):
    return float(r[k])


parser = argparse.ArgumentParser(
    description=(
        "Selectionne des coupures F3 pertinentes pour la reproduction materielle, "
        "en privilegiant les fautes tardives (donc les moins destructrices) "
        "qui provoquent encore des erreurs ou confusions critiques."
    )
)
parser.add_argument("--fc", default="f3_fc_sweep.csv")
parser.add_argument("--conv", default="f3_conv_sweep.csv")
parser.add_argument("--output", default="f3_candidats_materiel.csv")
args = parser.parse_args()

rows = read_csv(args.fc) + read_csv(args.conv)

# Exclure les cas triviaux :
# - k=0 : aucune accumulation réalisée
# - k=total : nominal
candidates = [
    r for r in rows
    if 0 < I(r, "cut_k") < I(r, "total_terms")
]

# ------------------------------------------------------------
# 1) Fautes critiques tardives
# ------------------------------------------------------------
critical = [
    r for r in candidates
    if I(r, "new_critical_count") > 0
]

critical_sorted = sorted(
    critical,
    key=lambda r: (
        -F(r, "completion_percent"),   # faute la plus tardive d'abord
        -I(r, "new_critical_count"),
        -I(r, "new_error_count"),
        -I(r, "prediction_changed_count"),
    )
)

# ------------------------------------------------------------
# 2) Si pas assez, compléter par nouvelles erreurs tardives
# ------------------------------------------------------------
new_errors = [
    r for r in candidates
    if I(r, "new_error_count") > 0
]

new_errors_sorted = sorted(
    new_errors,
    key=lambda r: (
        -F(r, "completion_percent"),
        -I(r, "new_error_count"),
        -I(r, "prediction_changed_count"),
    )
)

selected = []
seen = set()

def add_row(r, reason):
    key = (r["layer"], I(r, "cut_k"))
    if key in seen:
        return
    seen.add(key)

    selected.append({
        "rank": len(selected) + 1,
        "selection_reason": reason,
        "layer": r["layer"],
        "cut_k": I(r, "cut_k"),
        "total_terms": I(r, "total_terms"),
        "skipped_terms": I(r, "total_terms") - I(r, "cut_k"),
        "completion_percent": F(r, "completion_percent"),
        "fault_accuracy_percent": F(r, "fault_accuracy_percent"),
        "delta_accuracy_points": F(r, "delta_accuracy_points"),
        "prediction_changed_count": I(r, "prediction_changed_count"),
        "new_error_count": I(r, "new_error_count"),
        "recovered_error_count": I(r, "recovered_error_count"),
        "new_critical_count": I(r, "new_critical_count"),
    })

# Essayer d'avoir des candidats des deux couches.
for layer in ["FullyConnected", "Conv2D"]:
    layer_crit = [r for r in critical_sorted if r["layer"] == layer]
    for r in layer_crit[:3]:
        add_row(r, "late_critical")

# Puis compléter jusqu'à 10.
for r in critical_sorted:
    if len(selected) >= 10:
        break
    add_row(r, "late_critical")

for r in new_errors_sorted:
    if len(selected) >= 10:
        break
    add_row(r, "late_new_error")

# ------------------------------------------------------------
# Résumé par couche : dernière coupure encore perturbante
# ------------------------------------------------------------
print("===== F3 - CANDIDATS MATERIEL =====")

for layer in ["FullyConnected", "Conv2D"]:
    layer_rows = [r for r in candidates if r["layer"] == layer]

    changed = [
        r for r in layer_rows
        if I(r, "prediction_changed_count") > 0
    ]
    errors = [
        r for r in layer_rows
        if I(r, "new_error_count") > 0
    ]
    crits = [
        r for r in layer_rows
        if I(r, "new_critical_count") > 0
    ]

    def latest(group):
        if not group:
            return None
        return max(group, key=lambda r: F(r, "completion_percent"))

    latest_changed = latest(changed)
    latest_error = latest(errors)
    latest_crit = latest(crits)

    print()
    print(layer)

    if latest_changed:
        print(
            "  Derniere coupure changeant une prediction : "
            f"k={I(latest_changed,'cut_k')}/{I(latest_changed,'total_terms')} "
            f"({F(latest_changed,'completion_percent'):.2f}%) | "
            f"predΔ={I(latest_changed,'prediction_changed_count')}"
        )

    if latest_error:
        print(
            "  Derniere coupure creant une erreur         : "
            f"k={I(latest_error,'cut_k')}/{I(latest_error,'total_terms')} "
            f"({F(latest_error,'completion_percent'):.2f}%) | "
            f"newErr={I(latest_error,'new_error_count')}"
        )

    if latest_crit:
        print(
            "  Derniere coupure critique                  : "
            f"k={I(latest_crit,'cut_k')}/{I(latest_crit,'total_terms')} "
            f"({F(latest_crit,'completion_percent'):.2f}%) | "
            f"newCrit={I(latest_crit,'new_critical_count')}"
        )
    else:
        print(
            "  Aucune confusion critique tardive observee."
        )

print()
print("===== TOP CANDIDATS POUR LA CIBLE =====")

for r in selected:
    print(
        f"{r['rank']:2d}. "
        f"{r['layer']:15s} | "
        f"k={r['cut_k']:4d}/{r['total_terms']} | "
        f"skip={r['skipped_terms']:4d} | "
        f"completion={r['completion_percent']:6.2f}% | "
        f"acc={r['fault_accuracy_percent']:6.2f}% | "
        f"newErr={r['new_error_count']:3d} | "
        f"newCrit={r['new_critical_count']:3d}"
    )

if selected:
    with Path(args.output).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=selected[0].keys()
        )
        w.writeheader()
        w.writerows(selected)

    print()
    print("✅", args.output)
else:
    print()
    print("Aucun candidat perturbant trouve.")
