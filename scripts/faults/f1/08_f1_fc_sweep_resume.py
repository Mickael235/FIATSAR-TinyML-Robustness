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

FC_TENSOR_INDEX = 4
FC_OUT = 12
FC_IN = 4000
FC_WEIGHT_COUNT = FC_OUT * FC_IN

FIELDS = [
    "tensor_index", "output_class_id", "output_class",
    "weight_index", "local_weight_index", "bit",
    "old_value", "new_value",
    "n_eval",
    "baseline_accuracy_percent",
    "fault_accuracy_percent",
    "delta_accuracy_points",
    "predictions_changed",
    "scores_changed",
    "new_errors",
    "recovered_errors",
    "critical_total_faulty",
    "new_critical_confusions",
    "max_abs_score_diff",
]

def signed_int8(v):
    return v - 256 if v >= 128 else v

def critical_mask(y, pred):
    out = np.zeros(len(y), dtype=bool)
    for i in range(len(y)):
        t = LABELS[int(y[i])]
        p = LABELS[int(pred[i])]
        out[i] = (
            (t, p) in CRITICAL_PAIRS
            or (
                t not in ("_silence_", "_unknown_")
                and p == "_silence_"
            )
        )
    return out

def build_faulty_model(raw_model, tensor_index, weight_index, bit):
    model = schema_fb.Model.GetRootAsModel(raw_model, 0)
    obj = schema_fb.ModelT.InitFromObj(model)
    tensor = obj.subgraphs[0].tensors[tensor_index]

    if tensor.type != schema_fb.TensorType.INT8:
        raise TypeError(f"Tensor {tensor_index} n'est pas INT8")

    bidx = tensor.buffer
    data = bytearray(obj.buffers[bidx].data)

    old_u8 = data[weight_index]
    new_u8 = old_u8 ^ (1 << bit)
    data[weight_index] = new_u8
    obj.buffers[bidx].data = data

    builder = flatbuffers.Builder(0)
    off = obj.Pack(builder)
    builder.Finish(off, file_identifier=b"TFL3")
    return bytes(builder.Output()), signed_int8(old_u8), signed_int8(new_u8)

def run_model(model_content, Xq):
    interpreter = tf.lite.Interpreter(
        model_content=model_content,
        num_threads=1
    )
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    scores = np.empty((len(Xq), 12), dtype=np.int8)
    preds = np.empty(len(Xq), dtype=np.int64)

    for i in range(len(Xq)):
        interpreter.set_tensor(inp["index"], Xq[i:i+1])
        interpreter.invoke()
        s = interpreter.get_tensor(out["index"])[0].astype(np.int8, copy=True)
        scores[i] = s
        preds[i] = int(np.argmax(s))
    return scores, preds

def choose_balanced_subset(y, per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in range(12):
        idx = np.flatnonzero(y == cls)
        n = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=n, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(chosen)
    return chosen

def choose_fc_weights(weights_per_output, seed):
    """
    Échantillonnage stratifié :
    pour chacune des 12 sorties FC, tirer N poids parmi ses 4000 entrées.
    Retourne [(global_index, output_class, local_index), ...].
    """
    rng = np.random.default_rng(seed)
    selected = []

    for out_cls in range(FC_OUT):
        local = rng.choice(
            np.arange(FC_IN),
            size=weights_per_output,
            replace=False
        )
        for li in local:
            gi = out_cls * FC_IN + int(li)
            selected.append((gi, out_cls, int(li)))

    return selected

def evaluate_fault(raw_model, Xq, y, scores_base, preds_base,
                   weight_index, bit):
    faulty_raw, old_value, new_value = build_faulty_model(
        raw_model,
        FC_TENSOR_INDEX,
        weight_index,
        bit
    )

    scores_fault, preds_fault = run_model(faulty_raw, Xq)

    correct_base = preds_base == y
    correct_fault = preds_fault == y
    changed_pred = preds_fault != preds_base
    changed_scores = np.any(scores_fault != scores_base, axis=1)

    new_errors = (~correct_fault) & correct_base
    recovered = correct_fault & (~correct_base)

    base_crit = critical_mask(y, preds_base)
    fault_crit = critical_mask(y, preds_fault)
    new_crit = fault_crit & (~base_crit)

    diff = scores_fault.astype(np.int16) - scores_base.astype(np.int16)

    out_cls = weight_index // FC_IN
    local_idx = weight_index % FC_IN

    return {
        "tensor_index": FC_TENSOR_INDEX,
        "output_class_id": out_cls,
        "output_class": LABELS[out_cls],
        "weight_index": weight_index,
        "local_weight_index": local_idx,
        "bit": bit,
        "old_value": old_value,
        "new_value": new_value,
        "n_eval": len(y),
        "baseline_accuracy_percent": 100.0 * np.mean(correct_base),
        "fault_accuracy_percent": 100.0 * np.mean(correct_fault),
        "delta_accuracy_points": 100.0 * (
            np.mean(correct_fault) - np.mean(correct_base)
        ),
        "predictions_changed": int(changed_pred.sum()),
        "scores_changed": int(changed_scores.sum()),
        "new_errors": int(new_errors.sum()),
        "recovered_errors": int(recovered.sum()),
        "critical_total_faulty": int(fault_crit.sum()),
        "new_critical_confusions": int(new_crit.sum()),
        "max_abs_score_diff": int(np.max(np.abs(diff))),
    }

def load_done(path):
    done = set()
    if not path.exists():
        return done

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            done.add((int(r["weight_index"]), int(r["bit"])))
    return done

def append_row(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def load_rows(path):
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            x = dict(r)
            for k in [
                "tensor_index", "output_class_id", "weight_index",
                "local_weight_index", "bit", "old_value", "new_value",
                "n_eval", "predictions_changed", "scores_changed",
                "new_errors", "recovered_errors",
                "critical_total_faulty", "new_critical_confusions",
                "max_abs_score_diff"
            ]:
                x[k] = int(float(x[k]))
            for k in [
                "baseline_accuracy_percent",
                "fault_accuracy_percent",
                "delta_accuracy_points"
            ]:
                x[k] = float(x[k])
            rows.append(x)
    return rows

def rank_key(r):
    return (
        r["delta_accuracy_points"],
        -r["new_critical_confusions"],
        -r["predictions_changed"],
        -r["new_errors"],
    )

parser = argparse.ArgumentParser(
    description="Sweep F1 FullyConnected échantillonné avec reprise."
)
parser.add_argument("--stage", choices=["screen", "validate"], required=True)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--seed", type=int, default=321)

parser.add_argument(
    "--weights-per-output",
    type=int,
    default=20,
    help="Nombre de poids tirés par neurone de sortie FC (défaut 20)"
)
parser.add_argument(
    "--per-class",
    type=int,
    default=5,
    help="Nombre d'entrées de test par classe au screen (défaut 5)"
)
parser.add_argument(
    "--bits",
    default="7,6,5,4,3,2,1,0"
)
parser.add_argument(
    "--screen-csv",
    default="f1_fc_screen.csv"
)
parser.add_argument(
    "--top",
    type=int,
    default=50
)
parser.add_argument(
    "--validate-csv",
    default="f1_fc_top50_500.csv"
)

args = parser.parse_args()

raw_model = Path(args.model).read_bytes()
base = np.load(args.baseline)

Xq_all = base["X_int8"].astype(np.int8)
y_all = base["y"].astype(np.int64)
scores_all = base["scores_int8"].astype(np.int8)
preds_all = base["preds"].astype(np.int64)

if Xq_all.shape != (500, 1960):
    raise ValueError(f"Baseline inattendue : {Xq_all.shape}")

weights = choose_fc_weights(args.weights_per_output, args.seed)
weight_meta = {gi: (oc, li) for gi, oc, li in weights}

if args.stage == "screen":
    bits = [int(x.strip()) for x in args.bits.split(",")]
    if any(b < 0 or b > 7 for b in bits):
        raise ValueError("Bits autorisés : 0..7")

    subset = choose_balanced_subset(
        y_all,
        per_class=args.per_class,
        seed=args.seed
    )

    Xq = Xq_all[subset]
    y = y_all[subset]
    scores_base = scores_all[subset]
    preds_base = preds_all[subset]

    out_csv = Path(args.screen_csv)
    done = load_done(out_csv)

    jobs = [
        (gi, bit)
        for gi, _, _ in weights
        for bit in bits
        if (gi, bit) not in done
    ]

    total = len(weights) * len(bits)

    print("===== F1 FULLYCONNECTED - SCREEN =====")
    print("Poids FC totaux :", FC_WEIGHT_COUNT)
    print("Poids échantillonnés :", len(weights),
          f"({args.weights_per_output}/sortie × 12)")
    print("Échantillons :", len(subset),
          f"({args.per_class}/classe)")
    print("Bits :", bits)
    print("Total fautes prévu :", total)
    print("Déjà terminées :", len(done))
    print("Restantes :", len(jobs))
    print("CSV :", out_csv)
    print("Reprise automatique : OUI")
    print()

    t0 = time.time()

    for n, (widx, bit) in enumerate(jobs, 1):
        row = evaluate_fault(
            raw_model, Xq, y, scores_base, preds_base,
            widx, bit
        )
        append_row(out_csv, row)

        if n == 1 or n % 25 == 0 or n == len(jobs):
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed else 0.0
            remaining = len(jobs) - n
            eta = remaining / rate / 60.0 if rate else 0.0

            print(
                f"[{len(done)+n}/{total}] "
                f"class={row['output_class']:10s} "
                f"w={widx:5d} bit={bit} | "
                f"Δacc={row['delta_accuracy_points']:+6.2f} | "
                f"predΔ={row['predictions_changed']:2d} | "
                f"newErr={row['new_errors']:2d} | "
                f"newCrit={row['new_critical_confusions']:2d} | "
                f"ETA≈{eta:.1f} min"
            )

    print()
    print("✅ SCREEN FC terminé :", out_csv)

else:
    screen_csv = Path(args.screen_csv)
    if not screen_csv.exists():
        raise FileNotFoundError(
            f"{screen_csv} introuvable. Lance d'abord --stage screen."
        )

    screen_rows = load_rows(screen_csv)
    unique = {}
    for r in screen_rows:
        unique[(r["weight_index"], r["bit"])] = r

    ranked = sorted(unique.values(), key=rank_key)[:args.top]

    out_csv = Path(args.validate_csv)
    done = load_done(out_csv)

    jobs = [
        r for r in ranked
        if (r["weight_index"], r["bit"]) not in done
    ]

    print("===== F1 FULLYCONNECTED - VALIDATION 500 =====")
    print("Top demandé :", args.top)
    print("Déjà validées :", len(done))
    print("Restantes :", len(jobs))
    print("CSV :", out_csv)
    print()

    t0 = time.time()

    for n, src in enumerate(jobs, 1):
        row = evaluate_fault(
            raw_model,
            Xq_all, y_all, scores_all, preds_all,
            src["weight_index"], src["bit"]
        )
        append_row(out_csv, row)

        if n == 1 or n % 5 == 0 or n == len(jobs):
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed else 0.0
            remaining = len(jobs) - n
            eta = remaining / rate / 60.0 if rate else 0.0

            print(
                f"[{len(done)+n}/{args.top}] "
                f"class={row['output_class']:10s} "
                f"w={row['weight_index']:5d} bit={row['bit']} | "
                f"acc={row['fault_accuracy_percent']:6.2f}% | "
                f"Δ={row['delta_accuracy_points']:+6.2f} | "
                f"predΔ={row['predictions_changed']:3d} | "
                f"newErr={row['new_errors']:3d} | "
                f"newCrit={row['new_critical_confusions']:2d} | "
                f"ETA≈{eta:.1f} min"
            )

    rows = load_rows(out_csv)
    ranked = sorted(rows, key=rank_key)

    print()
    print("===== TOP 10 FC VALIDÉ SUR 500 =====")
    for r in ranked[:10]:
        print(
            f"FC class={r['output_class']:10s} "
            f"w={r['weight_index']:5d} bit={r['bit']} | "
            f"Δacc={r['delta_accuracy_points']:+.2f} pts | "
            f"predΔ={r['predictions_changed']:3d} | "
            f"newErr={r['new_errors']:3d} | "
            f"newCrit={r['new_critical_confusions']:2d}"
        )

    print()
    print("✅ VALIDATION FC terminée :", out_csv)
