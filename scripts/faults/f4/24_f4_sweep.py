import argparse
import csv
from pathlib import Path

import numpy as np


LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

CRITICAL_PAIRS = {
    ("stop", "go"), ("go", "stop"),
    ("left", "right"), ("right", "left"),
    ("on", "off"), ("off", "on"),
}


def is_critical(true_id, pred_id):
    t = LABELS[int(true_id)]
    p = LABELS[int(pred_id)]

    return int(
        (t, p) in CRITICAL_PAIRS
        or (
            t not in ("_silence_", "_unknown_")
            and p == "_silence_"
        )
    )


def metrics(y, base_preds, fault_preds):
    y = np.asarray(y)
    base_preds = np.asarray(base_preds)
    fault_preds = np.asarray(fault_preds)

    base_correct = base_preds == y
    fault_correct = fault_preds == y

    base_acc = 100.0 * np.mean(base_correct)
    fault_acc = 100.0 * np.mean(fault_correct)

    pred_changed = base_preds != fault_preds
    new_error = base_correct & (~fault_correct)
    recovered = (~base_correct) & fault_correct

    base_crit = np.array(
        [is_critical(t, p) for t, p in zip(y, base_preds)],
        dtype=bool
    )
    fault_crit = np.array(
        [is_critical(t, p) for t, p in zip(y, fault_preds)],
        dtype=bool
    )

    new_crit = (~base_crit) & fault_crit

    return {
        "baseline_accuracy_percent": base_acc,
        "fault_accuracy_percent": fault_acc,
        "delta_accuracy_points": fault_acc - base_acc,
        "prediction_changed_count": int(np.sum(pred_changed)),
        "prediction_changed_percent": 100.0 * np.mean(pred_changed),
        "new_error_count": int(np.sum(new_error)),
        "recovered_error_count": int(np.sum(recovered)),
        "new_critical_count": int(np.sum(new_crit)),
    }


def write_csv(path, rows):
    if not rows:
        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        w.writeheader()
        w.writerows(rows)


parser = argparse.ArgumentParser(
    description=(
        "F4 TinyConv : corruption de la couche de sortie. "
        "Mode A = argmax/softmax tronque apres k classes. "
        "Mode B = score d'une classe force au maximum int8."
    )
)

parser.add_argument(
    "--baseline",
    default="baseline_500.npz"
)
parser.add_argument(
    "--output-prefix",
    default="f4"
)
args = parser.parse_args()


base = np.load(args.baseline)

y = base["y"].astype(np.int64)
scores = base["scores_int8"].astype(np.int8)
base_preds = base["preds"].astype(np.int64)

if scores.shape != (len(y), 12):
    raise RuntimeError(
        f"Scores inattendus : {scores.shape}"
    )

n = len(y)

print("========================================")
print(" F4 - CORRUPTION COUCHE DE SORTIE")
print("========================================")
print("Echantillons fixes :", n)
print("Classes            :", len(LABELS))
print(
    "Accuracy nominale : "
    f"{100*np.mean(base_preds == y):.2f}%"
)
print()


# ============================================================
# F4-A : ARGMAX / SORTIE TRONQUEE
# ============================================================

print("===== F4-A : SORTIE TRONQUEE =====")
print(
    "Le calcul de sortie s'arrete apres les k premieres classes."
)

truncate_rows = []

for k in range(1, 13):
    fault_preds = np.argmax(
        scores[:, :k],
        axis=1
    ).astype(np.int64)

    m = metrics(
        y,
        base_preds,
        fault_preds
    )

    row = {
        "mode": "truncate",
        "cut_k": k,
        "total_outputs": 12,
        "completion_percent": 100.0 * k / 12.0,
        **m,
    }

    truncate_rows.append(row)

    print(
        f"k={k:2d}/12 | "
        f"acc={m['fault_accuracy_percent']:6.2f}% | "
        f"Δ={m['delta_accuracy_points']:+6.2f} | "
        f"predΔ={m['prediction_changed_count']:3d} | "
        f"newErr={m['new_error_count']:3d} | "
        f"newCrit={m['new_critical_count']:3d}"
    )

# k=12 doit reproduire le nominal.
nominal_row = truncate_rows[-1]

if (
    abs(
        float(
            nominal_row["delta_accuracy_points"]
        )
    ) > 1e-9
    or int(
        nominal_row["prediction_changed_count"]
    ) != 0
):
    raise RuntimeError(
        "Validation F4-A k=12 non nominale."
    )


# ============================================================
# F4-B : SCORE FORCE
# ============================================================

print()
print("===== F4-B : SCORE FORCE =====")
print(
    "Le score d'une classe est force a 127 avant l'argmax."
)

force_rows = []

for class_id in range(12):
    fault_scores = scores.copy()

    # 127 est la valeur max int8.
    fault_scores[:, class_id] = np.int8(127)

    fault_preds = np.argmax(
        fault_scores,
        axis=1
    ).astype(np.int64)

    m = metrics(
        y,
        base_preds,
        fault_preds
    )

    row = {
        "mode": "force_score_max",
        "forced_class_id": class_id,
        "forced_class": LABELS[class_id],
        "forced_value_int8": 127,
        **m,
    }

    force_rows.append(row)

    print(
        f"class={class_id:2d} "
        f"{LABELS[class_id]:12s} | "
        f"acc={m['fault_accuracy_percent']:6.2f}% | "
        f"Δ={m['delta_accuracy_points']:+6.2f} | "
        f"predΔ={m['prediction_changed_count']:3d} | "
        f"newErr={m['new_error_count']:3d} | "
        f"newCrit={m['new_critical_count']:3d}"
    )


# ============================================================
# TOP CANDIDATS
# ============================================================

candidates = []

for r in truncate_rows[:-1]:
    candidates.append({
        "mode": r["mode"],
        "parameter": f"k={int(r['cut_k'])}/12",
        "completion_percent": r["completion_percent"],
        "fault_accuracy_percent": r["fault_accuracy_percent"],
        "delta_accuracy_points": r["delta_accuracy_points"],
        "prediction_changed_count": r["prediction_changed_count"],
        "new_error_count": r["new_error_count"],
        "new_critical_count": r["new_critical_count"],
    })

for r in force_rows:
    candidates.append({
        "mode": r["mode"],
        "parameter": (
            f"class={int(r['forced_class_id'])}:"
            f"{r['forced_class']}"
        ),
        "completion_percent": "",
        "fault_accuracy_percent": r["fault_accuracy_percent"],
        "delta_accuracy_points": r["delta_accuracy_points"],
        "prediction_changed_count": r["prediction_changed_count"],
        "new_error_count": r["new_error_count"],
        "new_critical_count": r["new_critical_count"],
    })

ranked = sorted(
    candidates,
    key=lambda r: (
        -int(r["new_critical_count"]),
        -int(r["new_error_count"]),
        -int(r["prediction_changed_count"]),
        float(r["fault_accuracy_percent"]),
    )
)

top10 = []

for rank, r in enumerate(
    ranked[:10],
    1
):
    top10.append({
        "rank": rank,
        **r,
    })


# ============================================================
# CANDIDATS MATERIEL
# ============================================================

# Pour truncate : privilégier la coupure la plus tardive
# provoquant encore une confusion critique.
late_critical = [
    r for r in truncate_rows[:-1]
    if int(r["new_critical_count"]) > 0
]

late_error = [
    r for r in truncate_rows[:-1]
    if int(r["new_error_count"]) > 0
]

latest_critical = (
    max(
        late_critical,
        key=lambda r: int(r["cut_k"])
    )
    if late_critical
    else None
)

latest_error = (
    max(
        late_error,
        key=lambda r: int(r["cut_k"])
    )
    if late_error
    else None
)

# Pour score force : candidat provoquant le plus de critiques.
worst_force = max(
    force_rows,
    key=lambda r: (
        int(r["new_critical_count"]),
        int(r["new_error_count"]),
        int(r["prediction_changed_count"]),
    )
)


# ============================================================
# EXPORTS
# ============================================================

truncate_path = (
    f"{args.output_prefix}_truncate_sweep.csv"
)

force_path = (
    f"{args.output_prefix}_force_score_sweep.csv"
)

top10_path = (
    f"{args.output_prefix}_top10.csv"
)

write_csv(
    truncate_path,
    truncate_rows
)

write_csv(
    force_path,
    force_rows
)

write_csv(
    top10_path,
    top10
)


# ============================================================
# RESUME
# ============================================================

resume = []

if latest_critical is not None:
    resume.append({
        "selection": "latest_critical_truncate",
        "mode": "truncate",
        "parameter": (
            f"k={int(latest_critical['cut_k'])}/12"
        ),
        "fault_accuracy_percent": latest_critical[
            "fault_accuracy_percent"
        ],
        "delta_accuracy_points": latest_critical[
            "delta_accuracy_points"
        ],
        "prediction_changed_count": latest_critical[
            "prediction_changed_count"
        ],
        "new_error_count": latest_critical[
            "new_error_count"
        ],
        "new_critical_count": latest_critical[
            "new_critical_count"
        ],
    })

if latest_error is not None:
    resume.append({
        "selection": "latest_error_truncate",
        "mode": "truncate",
        "parameter": (
            f"k={int(latest_error['cut_k'])}/12"
        ),
        "fault_accuracy_percent": latest_error[
            "fault_accuracy_percent"
        ],
        "delta_accuracy_points": latest_error[
            "delta_accuracy_points"
        ],
        "prediction_changed_count": latest_error[
            "prediction_changed_count"
        ],
        "new_error_count": latest_error[
            "new_error_count"
        ],
        "new_critical_count": latest_error[
            "new_critical_count"
        ],
    })

resume.append({
    "selection": "worst_forced_score",
    "mode": "force_score_max",
    "parameter": (
        f"class={int(worst_force['forced_class_id'])}:"
        f"{worst_force['forced_class']}"
    ),
    "fault_accuracy_percent": worst_force[
        "fault_accuracy_percent"
    ],
    "delta_accuracy_points": worst_force[
        "delta_accuracy_points"
    ],
    "prediction_changed_count": worst_force[
        "prediction_changed_count"
    ],
    "new_error_count": worst_force[
        "new_error_count"
    ],
    "new_critical_count": worst_force[
        "new_critical_count"
    ],
})

resume_path = (
    f"{args.output_prefix}_resume.csv"
)

write_csv(
    resume_path,
    resume
)


print()
print("===== F4 - CANDIDATS MATERIEL =====")

if latest_critical is not None:
    print(
        "Derniere coupure encore critique : "
        f"k={int(latest_critical['cut_k'])}/12 | "
        f"acc={float(latest_critical['fault_accuracy_percent']):.2f}% | "
        f"newErr={int(latest_critical['new_error_count'])} | "
        f"newCrit={int(latest_critical['new_critical_count'])}"
    )
else:
    print(
        "Aucune coupure truncate ne cree de confusion critique."
    )

if latest_error is not None:
    print(
        "Derniere coupure creant une erreur : "
        f"k={int(latest_error['cut_k'])}/12 | "
        f"newErr={int(latest_error['new_error_count'])}"
    )

print(
    "Score force le plus critique : "
    f"{worst_force['forced_class']} | "
    f"acc={float(worst_force['fault_accuracy_percent']):.2f}% | "
    f"newErr={int(worst_force['new_error_count'])} | "
    f"newCrit={int(worst_force['new_critical_count'])}"
)

print()
print("===== TOP 10 F4 =====")

for r in top10:
    print(
        f"{int(r['rank']):2d}. "
        f"{r['mode']:15s} | "
        f"{r['parameter']:18s} | "
        f"acc={float(r['fault_accuracy_percent']):6.2f}% | "
        f"Δ={float(r['delta_accuracy_points']):+6.2f} | "
        f"newErr={int(r['new_error_count']):3d} | "
        f"newCrit={int(r['new_critical_count']):3d}"
    )

print()
print("✅", truncate_path)
print("✅", force_path)
print("✅", resume_path)
print("✅", top10_path)
