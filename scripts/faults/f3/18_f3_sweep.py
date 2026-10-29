import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import time
from pathlib import Path

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


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


def save_model_bytes(model_obj):
    builder = flatbuffers.Builder(0)
    offset = model_obj.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def evaluate_predictions(y, base_preds, fault_preds):
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

    base_crit = np.array([
        is_critical(t, p)
        for t, p in zip(y, base_preds)
    ], dtype=bool)

    fault_crit = np.array([
        is_critical(t, p)
        for t, p in zip(y, fault_preds)
    ], dtype=bool)

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
        "F3 TinyConv : simulation d'une sortie prematuree "
        "de boucle d'accumulation (k premieres contributions seulement)."
    )
)

parser.add_argument(
    "--model",
    default="model_original.tflite"
)
parser.add_argument(
    "--baseline",
    default="baseline_500.npz"
)
parser.add_argument(
    "--fc-tail",
    default="f2_tail_fc_softmax.tflite"
)
parser.add_argument(
    "--fc-step",
    type=int,
    default=25,
    help=(
        "Pas du balayage FC. 25 = rapide. "
        "1 = exhaustif 0..4000."
    )
)
parser.add_argument(
    "--output-prefix",
    default="f3"
)
args = parser.parse_args()


for p in [
    args.model,
    args.baseline,
    args.fc_tail,
]:
    if not Path(p).exists():
        raise FileNotFoundError(p)


# ============================================================
# BASELINE
# ============================================================

base = np.load(args.baseline)

Xq = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)
base_scores = base["scores_int8"].astype(np.int8)
base_preds = base["preds"].astype(np.int64)

n_samples = len(Xq)

print("========================================")
print(" F3 - TRONCATURE DE BOUCLE D'ACCUMULATION")
print("========================================")
print("Echantillons fixes :", n_samples)
print("FC step            :", args.fc_step)
print()


# ============================================================
# CACHE TENSOR 9 = ENTREE FC
# ============================================================

print("===== CACHE DES ACTIVATIONS FC =====")

full = tf.lite.Interpreter(
    model_path=args.model,
    experimental_preserve_all_tensors=True,
    num_threads=1,
)
full.allocate_tensors()

full_in = full.get_input_details()[0]
full_out = full.get_output_details()[0]

details = {
    d["index"]: d
    for d in full.get_tensor_details()
}

if 9 not in details:
    raise RuntimeError(
        "Tensor 9 introuvable."
    )

fc_input_detail = details[9]

if tuple(fc_input_detail["shape"]) != (1, 4000):
    raise RuntimeError(
        f"Tensor 9 inattendu : {fc_input_detail['shape']}"
    )

FC_TERMS = 4000
fc_zero_point = int(
    fc_input_detail["quantization"][1]
)

fc_acts = np.empty(
    (n_samples, FC_TERMS),
    dtype=np.int8
)

exact = 0

for i in range(n_samples):
    full.set_tensor(
        full_in["index"],
        Xq[i:i+1]
    )
    full.invoke()

    fc_acts[i] = full.get_tensor(9)[0]

    out = full.get_tensor(
        full_out["index"]
    )[0]

    if np.array_equal(
        out,
        base_scores[i]
    ):
        exact += 1

if exact != n_samples:
    raise RuntimeError(
        f"Baseline incoherente : {exact}/{n_samples}"
    )

print(
    f"Tensor 9 : [1,{FC_TERMS}] INT8 | "
    f"zero_point={fc_zero_point}"
)
print(
    f"Validation baseline : {exact}/{n_samples}"
)
print()


# ============================================================
# F3 - FULLY CONNECTED
# ============================================================

print("===== F3 FULLY CONNECTED =====")
print(
    "Interpretation : conserver les k premieres "
    "contributions, annuler les suivantes."
)

fc_tail = tf.lite.Interpreter(
    model_path=args.fc_tail,
    num_threads=1,
)
fc_tail.allocate_tensors()

fc_tail_in = fc_tail.get_input_details()[0]
fc_tail_out = fc_tail.get_output_details()[0]

if int(
    fc_tail_in["quantization"][1]
) != fc_zero_point:
    raise RuntimeError(
        "Zero-point FC tail incoherent."
    )

fc_cutpoints = list(
    range(
        0,
        FC_TERMS + 1,
        args.fc_step
    )
)

if fc_cutpoints[-1] != FC_TERMS:
    fc_cutpoints.append(
        FC_TERMS
    )

fc_rows = []

t0 = time.time()

for pos, k in enumerate(
    fc_cutpoints,
    1
):
    fault_preds = np.empty(
        n_samples,
        dtype=np.int64
    )

    for i in range(n_samples):
        a = fc_acts[i].copy()

        if k < FC_TERMS:
            a[k:] = np.int8(
                fc_zero_point
            )

        fc_tail.set_tensor(
            fc_tail_in["index"],
            a.reshape(
                fc_tail_in["shape"]
            )
        )
        fc_tail.invoke()

        scores = fc_tail.get_tensor(
            fc_tail_out["index"]
        )[0]

        fault_preds[i] = int(
            np.argmax(scores)
        )

    metrics = evaluate_predictions(
        y,
        base_preds,
        fault_preds
    )

    row = {
        "layer": "FullyConnected",
        "cut_k": k,
        "total_terms": FC_TERMS,
        "completion_percent": (
            100.0 * k / FC_TERMS
        ),
        **metrics,
    }

    fc_rows.append(row)

    if (
        pos == 1
        or pos % 10 == 0
        or pos == len(fc_cutpoints)
    ):
        print(
            f"[FC {pos:3d}/{len(fc_cutpoints)}] "
            f"k={k:4d}/{FC_TERMS} | "
            f"acc={metrics['fault_accuracy_percent']:6.2f}% | "
            f"Δ={metrics['delta_accuracy_points']:+6.2f} | "
            f"predΔ={metrics['prediction_changed_count']:3d} | "
            f"newErr={metrics['new_error_count']:3d} | "
            f"newCrit={metrics['new_critical_count']:3d}"
        )

print(
    f"FC termine en {time.time()-t0:.1f} s"
)
print()


# ============================================================
# F3 - CONV2D
# ============================================================

print("===== F3 CONV2D =====")
print(
    "Interpretation : dans chaque filtre, conserver "
    "les k premiers MAC du noyau et annuler les suivants."
)

raw = Path(
    args.model
).read_bytes()

root = schema_fb.Model.GetRootAsModel(
    raw,
    0
)

model_t = schema_fb.ModelT.InitFromObj(
    root
)

sg = model_t.subgraphs[0]

# Recherche du tensor poids Conv2D attendu [8,10,8,1].
conv_tensor_idx = None

for i, t in enumerate(sg.tensors):
    shape = tuple(int(x) for x in t.shape)

    if shape == (8, 10, 8, 1):
        conv_tensor_idx = i
        break

if conv_tensor_idx is None:
    raise RuntimeError(
        "Tensor de poids Conv2D [8,10,8,1] introuvable."
    )

conv_tensor = sg.tensors[
    conv_tensor_idx
]

conv_buffer_idx = int(
    conv_tensor.buffer
)

conv_buffer = model_t.buffers[
    conv_buffer_idx
]

conv_shape = tuple(
    int(x)
    for x in conv_tensor.shape
)

conv_weights = np.frombuffer(
    bytes(conv_buffer.data),
    dtype=np.int8
).copy().reshape(
    conv_shape
)

OUT_CHANNELS = conv_shape[0]
CONV_TERMS = int(
    np.prod(conv_shape[1:])
)

# Zero-points des poids par canal.
q = conv_tensor.quantization

if q is None or q.zeroPoint is None:
    conv_zp = np.zeros(
        OUT_CHANNELS,
        dtype=np.int8
    )
else:
    zps = np.asarray(
        q.zeroPoint,
        dtype=np.int64
    )

    if len(zps) == 1:
        conv_zp = np.full(
            OUT_CHANNELS,
            int(zps[0]),
            dtype=np.int8
        )
    elif len(zps) == OUT_CHANNELS:
        conv_zp = zps.astype(
            np.int8
        )
    else:
        raise RuntimeError(
            f"Zero-points Conv inattendus : {zps}"
        )

print(
    f"Poids Conv : tensor {conv_tensor_idx}, "
    f"shape={conv_shape}, "
    f"{CONV_TERMS} contributions/sortie"
)
print(
    "Zero-points poids Conv :",
    conv_zp.tolist()
)

conv_rows = []

t0 = time.time()

for k in range(
    0,
    CONV_TERMS + 1
):
    m = schema_fb.ModelT.InitFromObj(
        root
    )

    w = conv_weights.copy().reshape(
        OUT_CHANNELS,
        CONV_TERMS
    )

    if k < CONV_TERMS:
        for oc in range(
            OUT_CHANNELS
        ):
            w[oc, k:] = conv_zp[oc]

    m.buffers[
        conv_buffer_idx
    ].data = bytearray(
        w.reshape(conv_shape)
        .astype(np.int8)
        .tobytes()
    )

    model_bytes = save_model_bytes(
        m
    )

    interp = tf.lite.Interpreter(
        model_content=model_bytes,
        num_threads=1
    )
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    fault_preds = np.empty(
        n_samples,
        dtype=np.int64
    )

    for i in range(n_samples):
        interp.set_tensor(
            inp["index"],
            Xq[i:i+1]
        )
        interp.invoke()

        scores = interp.get_tensor(
            out["index"]
        )[0]

        fault_preds[i] = int(
            np.argmax(scores)
        )

    metrics = evaluate_predictions(
        y,
        base_preds,
        fault_preds
    )

    row = {
        "layer": "Conv2D",
        "cut_k": k,
        "total_terms": CONV_TERMS,
        "completion_percent": (
            100.0 * k / CONV_TERMS
        ),
        **metrics,
    }

    conv_rows.append(row)

    if (
        k == 0
        or k % 10 == 0
        or k == CONV_TERMS
    ):
        print(
            f"[Conv k={k:2d}/{CONV_TERMS}] "
            f"acc={metrics['fault_accuracy_percent']:6.2f}% | "
            f"Δ={metrics['delta_accuracy_points']:+6.2f} | "
            f"predΔ={metrics['prediction_changed_count']:3d} | "
            f"newErr={metrics['new_error_count']:3d} | "
            f"newCrit={metrics['new_critical_count']:3d}"
        )

print(
    f"Conv termine en {time.time()-t0:.1f} s"
)
print()


# ============================================================
# VALIDATION K = FULL
# ============================================================

fc_full = next(
    r for r in fc_rows
    if int(r["cut_k"]) == FC_TERMS
)

conv_full = next(
    r for r in conv_rows
    if int(r["cut_k"]) == CONV_TERMS
)

if (
    abs(
        float(
            fc_full["delta_accuracy_points"]
        )
    ) > 1e-9
    or int(
        fc_full["prediction_changed_count"]
    ) != 0
):
    raise RuntimeError(
        "Validation FC k=4000 non nominale."
    )

if (
    abs(
        float(
            conv_full["delta_accuracy_points"]
        )
    ) > 1e-9
    or int(
        conv_full["prediction_changed_count"]
    ) != 0
):
    raise RuntimeError(
        "Validation Conv k=80 non nominale."
    )


# ============================================================
# EXPORT CSV
# ============================================================

fc_path = (
    f"{args.output_prefix}_fc_sweep.csv"
)

conv_path = (
    f"{args.output_prefix}_conv_sweep.csv"
)

write_csv(
    fc_path,
    fc_rows
)

write_csv(
    conv_path,
    conv_rows
)


# ============================================================
# TOP 10 COUPURES
# ============================================================

all_rows = (
    fc_rows
    + conv_rows
)

# Exclut les points nominaux k=total_terms.
candidates = [
    r for r in all_rows
    if int(r["cut_k"])
    < int(r["total_terms"])
]

ranked = sorted(
    candidates,
    key=lambda r: (
        -int(r["new_critical_count"]),
        -int(r["new_error_count"]),
        -int(r["prediction_changed_count"]),
        float(r["fault_accuracy_percent"]),
        int(r["cut_k"]),
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

top_path = (
    f"{args.output_prefix}_top10.csv"
)

write_csv(
    top_path,
    top10
)


# ============================================================
# RESUME
# ============================================================

def worst_for(rows):
    non_nominal = [
        r for r in rows
        if int(r["cut_k"])
        < int(r["total_terms"])
    ]

    return sorted(
        non_nominal,
        key=lambda r: (
            -int(r["new_critical_count"]),
            -int(r["new_error_count"]),
            float(r["fault_accuracy_percent"]),
        )
    )[0]


worst_fc = worst_for(
    fc_rows
)

worst_conv = worst_for(
    conv_rows
)

resume_rows = [
    {
        "layer": "FullyConnected",
        "n_cutpoints": len(fc_rows),
        "sweep_step": args.fc_step,
        "worst_cut_k": worst_fc["cut_k"],
        "total_terms": FC_TERMS,
        "worst_completion_percent": worst_fc["completion_percent"],
        "worst_accuracy_percent": worst_fc["fault_accuracy_percent"],
        "worst_delta_accuracy_points": worst_fc["delta_accuracy_points"],
        "worst_prediction_changed_count": worst_fc["prediction_changed_count"],
        "worst_new_error_count": worst_fc["new_error_count"],
        "worst_new_critical_count": worst_fc["new_critical_count"],
    },
    {
        "layer": "Conv2D",
        "n_cutpoints": len(conv_rows),
        "sweep_step": 1,
        "worst_cut_k": worst_conv["cut_k"],
        "total_terms": CONV_TERMS,
        "worst_completion_percent": worst_conv["completion_percent"],
        "worst_accuracy_percent": worst_conv["fault_accuracy_percent"],
        "worst_delta_accuracy_points": worst_conv["delta_accuracy_points"],
        "worst_prediction_changed_count": worst_conv["prediction_changed_count"],
        "worst_new_error_count": worst_conv["new_error_count"],
        "worst_new_critical_count": worst_conv["new_critical_count"],
    },
]

resume_path = (
    f"{args.output_prefix}_resume.csv"
)

write_csv(
    resume_path,
    resume_rows
)


# ============================================================
# AFFICHAGE FINAL
# ============================================================

print("===== F3 - RESUME =====")

for r in resume_rows:
    print(
        f"{r['layer']:15s} | "
        f"worst k={int(r['worst_cut_k'])}/"
        f"{int(r['total_terms'])} | "
        f"completion={float(r['worst_completion_percent']):6.2f}% | "
        f"acc={float(r['worst_accuracy_percent']):6.2f}% | "
        f"Δ={float(r['worst_delta_accuracy_points']):+6.2f} | "
        f"predΔ={int(r['worst_prediction_changed_count']):3d} | "
        f"newErr={int(r['worst_new_error_count']):3d} | "
        f"newCrit={int(r['worst_new_critical_count']):3d}"
    )

print()
print("===== TOP 10 COUPURES F3 =====")

for r in top10:
    print(
        f"{int(r['rank']):2d}. "
        f"{r['layer']:15s} | "
        f"k={int(r['cut_k']):4d}/"
        f"{int(r['total_terms'])} | "
        f"{float(r['completion_percent']):6.2f}% | "
        f"acc={float(r['fault_accuracy_percent']):6.2f}% | "
        f"Δ={float(r['delta_accuracy_points']):+6.2f} | "
        f"newErr={int(r['new_error_count']):3d} | "
        f"newCrit={int(r['new_critical_count']):3d}"
    )

print()
print("✅", fc_path)
print("✅", conv_path)
print("✅", resume_path)
print("✅", top_path)
print()
print(
    "Note : cette campagne F3 simule une sortie prematuree "
    "de boucle en ne conservant que les k premieres contributions. "
    "Le biais reste charge."
)
