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

# tiny_conv original, vérifié avec 03_inspect_tflite_weights.py
CONV_TENSOR_INDEX = 6
CONV_WEIGHT_COUNT = 640

FIELDS = [
    "tensor_index", "weight_index", "bit",
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
        true_name = LABELS[int(y[i])]
        pred_name = LABELS[int(pred[i])]
        out[i] = (
            (true_name, pred_name) in CRITICAL_PAIRS
            or (
                true_name not in ("_silence_", "_unknown_")
                and pred_name == "_silence_"
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
    faulty_raw = bytes(builder.Output())

    return faulty_raw, signed_int8(old_u8), signed_int8(new_u8)

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

def evaluate_fault(raw_model, Xq, y, scores_base, preds_base,
                   tensor_index, weight_index, bit):
    faulty_raw, old_value, new_value = build_faulty_model(
        raw_model, tensor_index, weight_index, bit
    )
    scores_fault, preds_fault = run_model(faulty_raw, Xq)

    correct_base = preds_base == y
    correct_fault = preds_fault == y
    pred_changed = preds_fault != preds_base
    scores_changed = np.any(scores_fault != scores_base, axis=1)

    new_errors = (~correct_fault) & correct_base
    recovered = correct_fault & (~correct_base)

    base_critical = critical_mask(y, preds_base)
    fault_critical = critical_mask(y, preds_fault)
    new_critical = fault_critical & (~base_critical)

    score_diff = (
        scores_fault.astype(np.int16)
        - scores_base.astype(np.int16)
    )

    base_acc = float(np.mean(correct_base))
    fault_acc = float(np.mean(correct_fault))

    return {
        "tensor_index": tensor_index,
        "weight_index": weight_index,
        "bit": bit,
        "old_value": old_value,
        "new_value": new_value,
        "n_eval": len(y),
        "baseline_accuracy_percent": 100.0 * base_acc,
        "fault_accuracy_percent": 100.0 * fault_acc,
        "delta_accuracy_points": 100.0 * (fault_acc - base_acc),
        "predictions_changed": int(pred_changed.sum()),
        "scores_changed": int(scores_changed.sum()),
        "new_errors": int(new_errors.sum()),
        "recovered_errors": int(recovered.sum()),
        "critical_total_faulty": int(fault_critical.sum()),
        "new_critical_confusions": int(new_critical.sum()),
        "max_abs_score_diff": int(np.max(np.abs(score_diff))),
    }

def load_done(csv_path):
    done = set()
    if not csv_path.exists():
        return done

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            done.add((
                int(row["weight_index"]),
                int(row["bit"])
            ))
    return done

def append_row(csv_path, row):
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def load_rows(csv_path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            x = {}
            for k, v in r.items():
                if k in ("baseline_accuracy_percent",
                         "fault_accuracy_percent",
                         "delta_accuracy_points"):
                    x[k] = float(v)
                else:
                    x[k] = int(float(v))
            rows.append(x)
    return rows

def rank_key(r):
    # Priorité : baisse accuracy, nouvelles confusions critiques,
    # changements de prédiction, nouvelles erreurs.
    return (
        r["delta_accuracy_points"],
        -r["new_critical_confusions"],
        -r["predictions_changed"],
        -r["new_errors"],
    )

parser = argparse.ArgumentParser(
    description="Sweep F1 Conv2D avec reprise automatique."
)
parser.add_argument(
    "--stage",
    choices=["screen", "validate"],
    required=True
)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--seed", type=int, default=123)

# SCREEN
parser.add_argument(
    "--per-class",
    type=int,
    default=5,
    help="Screen: nombre d'échantillons par classe (défaut 5 = 60)"
)
parser.add_argument(
    "--bits",
    default="7,6,5,4,3,2,1,0",
    help="Bits à tester (défaut: les 8 bits)"
)
parser.add_argument(
    "--screen-csv",
    default="f1_conv_screen.csv"
)

# VALIDATE
parser.add_argument(
    "--top",
    type=int,
    default=50,
    help="Validate: nombre de meilleures fautes à retester sur 500"
)
parser.add_argument(
    "--validate-csv",
    default="f1_conv_top50_500.csv"
)

args = parser.parse_args()

raw_model = Path(args.model).read_bytes()
base = np.load(args.baseline)

Xq_all = base["X_int8"].astype(np.int8)
y_all = base["y"].astype(np.int64)
scores_all = base["scores_int8"].astype(np.int8)
preds_all = base["preds"].astype(np.int64)

if Xq_all.shape != (500, 1960):
    raise ValueError(f"Baseline attendue (500,1960), reçue {Xq_all.shape}")

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
        (w, b)
        for w in range(CONV_WEIGHT_COUNT)
        for b in bits
        if (w, b) not in done
    ]

    total_full = CONV_WEIGHT_COUNT * len(bits)
    print("===== F1 CONV2D - SCREEN =====")
    print("Échantillons :", len(subset),
          f"({args.per_class}/classe)")
    print("Poids Conv2D :", CONV_WEIGHT_COUNT)
    print("Bits :", bits)
    print("Total fautes prévu :", total_full)
    print("Déjà terminées :", len(done))
    print("Restantes :", len(jobs))
    print("CSV :", out_csv)
    print("Reprise automatique : OUI")
    print()

    t0 = time.time()

    for n, (widx, bit) in enumerate(jobs, start=1):
        row = evaluate_fault(
            raw_model, Xq, y, scores_base, preds_base,
            CONV_TENSOR_INDEX, widx, bit
        )
        append_row(out_csv, row)

        done_now = len(done) + n

        if n == 1 or n % 25 == 0 or n == len(jobs):
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0.0
            remaining = len(jobs) - n
            eta_min = remaining / rate / 60.0 if rate > 0 else 0.0

            print(
                f"[{done_now}/{total_full}] "
                f"w={widx:3d} bit={bit} | "
                f"Δacc={row['delta_accuracy_points']:+6.2f} | "
                f"predΔ={row['predictions_changed']:2d} | "
                f"newErr={row['new_errors']:2d} | "
                f"newCrit={row['new_critical_confusions']:2d} | "
                f"ETA≈{eta_min:.1f} min"
            )

    print()
    print("✅ SCREEN terminé :", out_csv)
    print("Tu peux interrompre avec Ctrl+C : relancer la même commande reprend où elle s'est arrêtée.")

elif args.stage == "validate":
    screen_csv = Path(args.screen_csv)
    if not screen_csv.exists():
        raise FileNotFoundError(
            f"{screen_csv} introuvable. Lance d'abord --stage screen."
        )

    screen_rows = load_rows(screen_csv)

    # Une ligne unique par (poids,bit)
    unique = {}
    for r in screen_rows:
        unique[(r["weight_index"], r["bit"])] = r

    ranked = sorted(unique.values(), key=rank_key)
    selected = ranked[:args.top]

    out_csv = Path(args.validate_csv)
    done = load_done(out_csv)

    selected = [
        r for r in selected
        if (r["weight_index"], r["bit"]) not in done
    ]

    print("===== F1 CONV2D - VALIDATION 500 =====")
    print("Top demandé :", args.top)
    print("Déjà validées :", len(done))
    print("Restantes :", len(selected))
    print("CSV :", out_csv)
    print()

    t0 = time.time()

    for n, src in enumerate(selected, start=1):
        widx = src["weight_index"]
        bit = src["bit"]

        row = evaluate_fault(
            raw_model,
            Xq_all, y_all, scores_all, preds_all,
            CONV_TENSOR_INDEX, widx, bit
        )
        append_row(out_csv, row)

        if n == 1 or n % 5 == 0 or n == len(selected):
            elapsed = time.time() - t0
            rate = n / elapsed if elapsed > 0 else 0.0
            remaining = len(selected) - n
            eta_min = remaining / rate / 60.0 if rate > 0 else 0.0

            print(
                f"[{len(done)+n}/{args.top}] "
                f"w={widx:3d} bit={bit} | "
                f"acc={row['fault_accuracy_percent']:6.2f}% | "
                f"Δ={row['delta_accuracy_points']:+6.2f} | "
                f"predΔ={row['predictions_changed']:3d} | "
                f"newErr={row['new_errors']:3d} | "
                f"newCrit={row['new_critical_confusions']:2d} | "
                f"ETA≈{eta_min:.1f} min"
            )

    final_rows = load_rows(out_csv)
    final_ranked = sorted(final_rows, key=rank_key)

    print()
    print("===== TOP 10 VALIDÉ SUR 500 =====")
    for r in final_ranked[:10]:
        print(
            f"Conv2D w={r['weight_index']:3d} bit={r['bit']} | "
            f"Δacc={r['delta_accuracy_points']:+.2f} pts | "
            f"predΔ={r['predictions_changed']:3d} | "
            f"newErr={r['new_errors']:3d} | "
            f"newCrit={r['new_critical_confusions']:2d}"
        )

    print()
    print("✅ VALIDATION terminée :", out_csv)
