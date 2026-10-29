import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

CRITICAL_PAIRS = {
    ("stop", "go"), ("go", "stop"),
    ("left", "right"), ("right", "left"),
    ("on", "off"), ("off", "on"),
}

LAYERS = {
    "conv2d": {
        "display": "Conv2D_output_flat",
        "tensor": 9,
        "length": 4000,
        "tail": "fc",
    },
    "fc": {
        "display": "FullyConnected_logits",
        "tensor": 10,
        "length": 12,
        "tail": "softmax",
    },
}

DETAIL_FIELDS = [
    "mode", "layer", "tensor_index", "trial",
    "sample_order", "dataset_index",
    "true_id", "true_label",
    "activation_index", "bit",
    "old_int8", "new_int8",
    "scale", "zero_point",
    "old_real", "new_real",
    "base_pred_id", "base_pred",
    "fault_pred_id", "fault_pred",
    "base_correct", "fault_correct",
    "prediction_changed", "scores_changed",
    "new_error", "recovered_error",
    "base_critical", "fault_critical",
    "new_critical_confusion",
    "max_abs_score_diff",
    "true_class_score_delta",
]

def critical(true_id, pred_id):
    t = LABELS[int(true_id)]
    p = LABELS[int(pred_id)]
    return int(
        (t, p) in CRITICAL_PAIRS
        or (
            t not in ("_silence_", "_unknown_")
            and p == "_silence_"
        )
    )

def flip_int8(value, bit):
    u = np.array([value], dtype=np.int8).view(np.uint8)[0]
    u2 = np.uint8(int(u) ^ (1 << int(bit)))
    return int(np.array([u2], dtype=np.uint8).view(np.int8)[0])

def append_row(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def load_done(path):
    done = set()
    if not path.exists():
        return done
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            done.add((r["mode"], r["layer"], int(r["trial"])))
    return done

def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def make_schedule(mode, layer_key, n_trials, seed, n_samples):
    # Seeds séparées mais déterministes pour reprise parfaite.
    mode_offset = 0 if mode == "bitflip" else 100_000
    layer_offset = 0 if layer_key == "conv2d" else 10_000
    rng = np.random.default_rng(seed + mode_offset + layer_offset)

    info = LAYERS[layer_key]
    jobs = []

    for trial in range(n_trials):
        sample = int(rng.integers(0, n_samples))
        activation_index = int(rng.integers(0, info["length"]))
        bit = int(rng.integers(0, 8)) if mode == "bitflip" else -1
        jobs.append((trial, sample, activation_index, bit))

    return jobs

def write_summaries(detail_path, prefix):
    rows = read_rows(detail_path)

    def I(r, k):
        return int(float(r[k]))

    def F(r, k):
        return float(r[k])

    # --------------------------------------------------------
    # Résumé couche × mode
    # --------------------------------------------------------
    resume = []
    pairs = sorted(set((r["mode"], r["layer"]) for r in rows))

    for mode, layer in pairs:
        s = [r for r in rows if r["mode"] == mode and r["layer"] == layer]
        n = len(s)
        base_acc = 100.0 * np.mean([I(r, "base_correct") for r in s])
        fault_acc = 100.0 * np.mean([I(r, "fault_correct") for r in s])

        resume.append({
            "mode": mode,
            "layer": layer,
            "n_trials": n,
            "baseline_correct_percent_on_sampled_trials": base_acc,
            "faulty_correct_percent_on_sampled_trials": fault_acc,
            "delta_accuracy_points_monte_carlo": fault_acc - base_acc,
            "prediction_changed_count": sum(I(r, "prediction_changed") for r in s),
            "prediction_changed_percent": 100.0 * np.mean(
                [I(r, "prediction_changed") for r in s]
            ),
            "new_error_count": sum(I(r, "new_error") for r in s),
            "new_error_percent": 100.0 * np.mean(
                [I(r, "new_error") for r in s]
            ),
            "recovered_error_count": sum(I(r, "recovered_error") for r in s),
            "new_critical_count": sum(I(r, "new_critical_confusion") for r in s),
            "scores_changed_percent": 100.0 * np.mean(
                [I(r, "scores_changed") for r in s]
            ),
            "mean_max_abs_score_diff": float(np.mean(
                [F(r, "max_abs_score_diff") for r in s]
            )),
            "max_abs_score_diff": max(I(r, "max_abs_score_diff") for r in s),
        })

    resume_path = Path(f"{prefix}_resume.csv")
    with resume_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=resume[0].keys())
        w.writeheader()
        w.writerows(resume)

    # --------------------------------------------------------
    # Carte couche × bit, seulement bitflip
    # --------------------------------------------------------
    bit_rows = []
    bitflip = [r for r in rows if r["mode"] == "bitflip"]

    for layer in sorted(set(r["layer"] for r in bitflip)):
        for bit in range(7, -1, -1):
            s = [
                r for r in bitflip
                if r["layer"] == layer and I(r, "bit") == bit
            ]
            if not s:
                continue

            bit_rows.append({
                "layer": layer,
                "bit": bit,
                "n_trials": len(s),
                "prediction_changed_count": sum(
                    I(r, "prediction_changed") for r in s
                ),
                "prediction_changed_percent": 100.0 * np.mean(
                    [I(r, "prediction_changed") for r in s]
                ),
                "new_error_count": sum(I(r, "new_error") for r in s),
                "new_error_percent": 100.0 * np.mean(
                    [I(r, "new_error") for r in s]
                ),
                "new_critical_count": sum(
                    I(r, "new_critical_confusion") for r in s
                ),
                "scores_changed_percent": 100.0 * np.mean(
                    [I(r, "scores_changed") for r in s]
                ),
                "mean_max_abs_score_diff": float(np.mean(
                    [F(r, "max_abs_score_diff") for r in s]
                )),
                "max_abs_score_diff": max(
                    I(r, "max_abs_score_diff") for r in s
                ),
            })

    bit_path = Path(f"{prefix}_sensibilite_layer_bit.csv")
    if bit_rows:
        with bit_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=bit_rows[0].keys())
            w.writeheader()
            w.writerows(bit_rows)

    # --------------------------------------------------------
    # Top 10 événements F2
    # --------------------------------------------------------
    ranked = sorted(
        rows,
        key=lambda r: (
            -I(r, "new_critical_confusion"),
            -I(r, "new_error"),
            -I(r, "prediction_changed"),
            -I(r, "max_abs_score_diff"),
        )
    )

    top10_path = Path(f"{prefix}_top10.csv")
    with top10_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "rank", "mode", "layer", "tensor_index", "trial",
            "sample_order", "dataset_index",
            "true_label", "activation_index", "bit",
            "old_int8", "new_int8",
            "base_pred", "fault_pred",
            "new_error", "new_critical_confusion",
            "max_abs_score_diff",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for rank, r in enumerate(ranked[:10], 1):
            w.writerow({
                "rank": rank,
                **{k: r.get(k, "") for k in fields if k != "rank"}
            })

    return resume, ranked[:10], resume_path, bit_path, top10_path


parser = argparse.ArgumentParser(
    description="Campagne F2 : corruption transitoire d'activations INT8 TinyConv."
)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--fc-tail", default="f2_tail_fc_softmax.tflite")
parser.add_argument("--softmax-tail", default="f2_tail_softmax.tflite")
parser.add_argument(
    "--mode",
    choices=["bitflip", "zero", "both"],
    default="bitflip"
)
parser.add_argument(
    "--trials-per-layer",
    type=int,
    default=1000
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="f2_campaign.csv")
args = parser.parse_args()

for p in [args.model, args.baseline, args.fc_tail, args.softmax_tail]:
    if not Path(p).exists():
        raise FileNotFoundError(p)

base = np.load(args.baseline)
Xq = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)
indices = base["indices"].astype(np.int64)
scores_base_saved = base["scores_int8"].astype(np.int8)
preds_base_saved = base["preds"].astype(np.int64)

n_samples = len(Xq)

# ------------------------------------------------------------
# 1) Cache des activations nominales sur les 500 entrées
# ------------------------------------------------------------

print("===== F2 - PREPARATION DES ACTIVATIONS =====")

full = tf.lite.Interpreter(
    model_path=args.model,
    experimental_preserve_all_tensors=True,
    num_threads=1,
)
full.allocate_tensors()

full_in = full.get_input_details()[0]
full_out = full.get_output_details()[0]
details = {d["index"]: d for d in full.get_tensor_details()}

for idx in [9, 10]:
    if idx not in details:
        raise RuntimeError(f"Tensor {idx} introuvable.")

act9 = np.empty((n_samples, 4000), dtype=np.int8)
act10 = np.empty((n_samples, 12), dtype=np.int8)

exact_saved = 0

t0 = time.time()
for i in range(n_samples):
    full.set_tensor(full_in["index"], Xq[i:i+1])
    full.invoke()

    act9[i] = full.get_tensor(9)[0]
    act10[i] = full.get_tensor(10)[0]
    out = full.get_tensor(full_out["index"])[0]

    if np.array_equal(out, scores_base_saved[i]):
        exact_saved += 1

    if (i + 1) % 100 == 0:
        print(f"Cache activations : {i+1}/{n_samples}")

if exact_saved != n_samples:
    raise RuntimeError(
        f"Baseline incohérente : {exact_saved}/{n_samples} sorties exactes."
    )

print(f"Baseline vérifiée : {exact_saved}/{n_samples}")
print(f"Préparation : {time.time() - t0:.2f} s")

# Quantification des activations.
q9 = details[9]["quantization"]
q10 = details[10]["quantization"]

# ------------------------------------------------------------
# 2) Interpréteurs des tails
# ------------------------------------------------------------

tail_fc = tf.lite.Interpreter(
    model_path=args.fc_tail,
    num_threads=1,
)
tail_fc.allocate_tensors()
fc_in = tail_fc.get_input_details()[0]
fc_out = tail_fc.get_output_details()[0]

tail_sm = tf.lite.Interpreter(
    model_path=args.softmax_tail,
    num_threads=1,
)
tail_sm.allocate_tensors()
sm_in = tail_sm.get_input_details()[0]
sm_out = tail_sm.get_output_details()[0]

# ------------------------------------------------------------
# 3) Campagne
# ------------------------------------------------------------

modes = ["bitflip", "zero"] if args.mode == "both" else [args.mode]
out_path = Path(args.output)
done = load_done(out_path)

all_jobs = []
for mode in modes:
    for layer_key in ["conv2d", "fc"]:
        schedule = make_schedule(
            mode,
            layer_key,
            args.trials_per_layer,
            args.seed,
            n_samples,
        )
        for trial, sample, aidx, bit in schedule:
            all_jobs.append(
                (mode, layer_key, trial, sample, aidx, bit)
            )

remaining = [
    j for j in all_jobs
    if (j[0], LAYERS[j[1]]["display"], j[2]) not in done
]

print()
print("===== F2 - CAMPAGNE =====")
print("Mode               :", args.mode)
print("Essais / couche    :", args.trials_per_layer)
print("Couches            : 2")
print("Total planifié     :", len(all_jobs))
print("Déjà présents      :", len(done))
print("Restants           :", len(remaining))
print("Seed               :", args.seed)
print("Sortie             :", out_path)
print()

start = time.time()

for n, (mode, layer_key, trial, sample, aidx, bit) in enumerate(remaining, 1):
    info = LAYERS[layer_key]
    layer_name = info["display"]

    if layer_key == "conv2d":
        activation = act9[sample].copy()
        scale, zp = q9
        tail = tail_fc
        tail_in = fc_in
        tail_out = fc_out
    else:
        activation = act10[sample].copy()
        scale, zp = q10
        tail = tail_sm
        tail_in = sm_in
        tail_out = sm_out

    old = int(activation[aidx])

    if mode == "bitflip":
        new = flip_int8(old, bit)
    else:
        # En int8 quantifié, la valeur réelle 0 est le zero-point.
        new = int(zp)

    activation[aidx] = np.int8(new)

    tail.set_tensor(tail_in["index"], activation.reshape(tail_in["shape"]))
    tail.invoke()
    fault_scores = tail.get_tensor(tail_out["index"])[0].astype(np.int8, copy=True)

    base_scores = scores_base_saved[sample]
    base_pred = int(preds_base_saved[sample])
    fault_pred = int(np.argmax(fault_scores))
    true_id = int(y[sample])

    base_correct = int(base_pred == true_id)
    fault_correct = int(fault_pred == true_id)
    pred_changed = int(base_pred != fault_pred)
    scores_changed = int(not np.array_equal(base_scores, fault_scores))

    base_crit = critical(true_id, base_pred)
    fault_crit = critical(true_id, fault_pred)

    diff = (
        fault_scores.astype(np.int16)
        - base_scores.astype(np.int16)
    )

    row = {
        "mode": mode,
        "layer": layer_name,
        "tensor_index": info["tensor"],
        "trial": trial,
        "sample_order": sample,
        "dataset_index": int(indices[sample]),
        "true_id": true_id,
        "true_label": LABELS[true_id],
        "activation_index": aidx,
        "bit": bit if mode == "bitflip" else -1,
        "old_int8": old,
        "new_int8": new,
        "scale": float(scale),
        "zero_point": int(zp),
        "old_real": float(scale * (old - zp)),
        "new_real": float(scale * (new - zp)),
        "base_pred_id": base_pred,
        "base_pred": LABELS[base_pred],
        "fault_pred_id": fault_pred,
        "fault_pred": LABELS[fault_pred],
        "base_correct": base_correct,
        "fault_correct": fault_correct,
        "prediction_changed": pred_changed,
        "scores_changed": scores_changed,
        "new_error": int(base_correct == 1 and fault_correct == 0),
        "recovered_error": int(base_correct == 0 and fault_correct == 1),
        "base_critical": base_crit,
        "fault_critical": fault_crit,
        "new_critical_confusion": int(base_crit == 0 and fault_crit == 1),
        "max_abs_score_diff": int(np.max(np.abs(diff))),
        "true_class_score_delta": int(diff[true_id]),
    }

    append_row(out_path, row)

    if n == 1 or n % 100 == 0 or n == len(remaining):
        elapsed = time.time() - start
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = (len(remaining) - n) / rate if rate > 0 else 0.0

        print(
            f"[{len(done)+n:4d}/{len(all_jobs)}] "
            f"{mode:7s} | {layer_name:22s} | "
            f"trial={trial:4d} | sample={sample:3d} | "
            f"idx={aidx:4d} | "
            f"bit={bit if mode == 'bitflip' else '-'} | "
            f"{old:4d}->{new:4d} | "
            f"predΔ={pred_changed} | "
            f"newErr={row['new_error']} | "
            f"newCrit={row['new_critical_confusion']} | "
            f"ETA≈{eta:.1f}s"
        )

# ------------------------------------------------------------
# 4) Synthèse
# ------------------------------------------------------------

prefix = str(out_path.with_suffix(""))
resume, top10, resume_path, bit_path, top10_path = write_summaries(
    out_path,
    prefix,
)

print()
print("===== RESUME F2 =====")
for r in resume:
    print(
        f"{r['mode']:7s} | {r['layer']:22s} | "
        f"N={r['n_trials']:4d} | "
        f"Δacc_MC={r['delta_accuracy_points_monte_carlo']:+6.2f} pts | "
        f"predΔ={r['prediction_changed_percent']:6.2f}% | "
        f"newErr={r['new_error_percent']:6.2f}% | "
        f"newCrit={r['new_critical_count']}"
    )

print()
print("===== TOP 10 EVENEMENTS F2 =====")
for i, r in enumerate(top10, 1):
    print(
        f"{i:2d}. {r['mode']:7s} | "
        f"{r['layer']:22s} | "
        f"sample={int(r['sample_order']):3d} | "
        f"idx={int(r['activation_index']):4d} | "
        f"bit={r['bit']} | "
        f"{r['old_int8']}->{r['new_int8']} | "
        f"{r['base_pred']}->{r['fault_pred']} | "
        f"newErr={r['new_error']} | "
        f"newCrit={r['new_critical_confusion']} | "
        f"scoreΔmax={r['max_abs_score_diff']}"
    )

print()
print("✅", out_path)
print("✅", resume_path)
if Path(bit_path).exists():
    print("✅", bit_path)
print("✅", top10_path)
print()
print("Note : la mise à zéro quantifiée utilise le zero-point du tensor,")
print("pas la valeur entière 0.")
