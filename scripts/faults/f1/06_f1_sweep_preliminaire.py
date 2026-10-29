import argparse
import csv
import hashlib

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

CRITICAL_PAIRS = {
    ("stop", "go"),
    ("go", "stop"),
    ("left", "right"),
    ("right", "left"),
    ("on", "off"),
    ("off", "on"),
}

def signed_int8(u):
    return u - 256 if u >= 128 else u

def is_critical(true_id, pred_id):
    true_name = LABELS[int(true_id)]
    pred_name = LABELS[int(pred_id)]
    return (
        (true_name, pred_name) in CRITICAL_PAIRS
        or (
            true_name not in ("_silence_", "_unknown_")
            and pred_name == "_silence_"
        )
    )

def build_faulty_model(raw_model, tensor_index, weight_index, bit):
    model = schema_fb.Model.GetRootAsModel(raw_model, 0)
    obj = schema_fb.ModelT.InitFromObj(model)

    sg = obj.subgraphs[0]
    tensor = sg.tensors[tensor_index]

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

    name = tensor.name
    if isinstance(name, (bytes, bytearray)):
        name = name.decode("utf-8", errors="replace")

    return faulty_raw, {
        "tensor_index": tensor_index,
        "tensor_name": str(name),
        "buffer_index": bidx,
        "weight_index": weight_index,
        "bit": bit,
        "old_value": signed_int8(old_u8),
        "new_value": signed_int8(new_u8),
        "old_hex": f"0x{old_u8:02X}",
        "new_hex": f"0x{new_u8:02X}",
    }

def run_model(model_content, Xq):
    interpreter = tf.lite.Interpreter(model_content=model_content)
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

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--output", default="f1_sweep_preliminaire.csv")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--samples-per-tensor",
    type=int,
    default=5,
    help="Nombre de poids tirés dans chaque tenseur (défaut 5)"
)
parser.add_argument(
    "--bits",
    default="7,6,4,0",
    help="Bits testés, ex. 7,6,4,0"
)
args = parser.parse_args()

bits = [int(x) for x in args.bits.split(",")]
for b in bits:
    if b < 0 or b > 7:
        raise ValueError("Chaque bit doit être compris entre 0 et 7")

raw_model = open(args.model, "rb").read()
baseline = np.load(args.baseline)

Xq = baseline["X_int8"].astype(np.int8)
y = baseline["y"].astype(np.int64)
scores_base = baseline["scores_int8"].astype(np.int8)
preds_base = baseline["preds"].astype(np.int64)

if Xq.shape != (500, 1960):
    raise ValueError(f"Baseline inattendue : {Xq.shape}")

correct_base = preds_base == y
base_acc = float(np.mean(correct_base))
base_critical = np.array(
    [is_critical(y[i], preds_base[i]) for i in range(len(y))],
    dtype=bool
)

# Tenseurs F1 connus du tiny_conv inspecté
targets = [
    # tensor_index, nombre de poids, nom court
    (4, 48000, "FullyConnected"),
    (6, 640, "Conv2D"),
]

rng = np.random.default_rng(args.seed)
faults = []

for tensor_index, nweights, short_name in targets:
    if args.samples_per_tensor > nweights:
        raise ValueError("samples-per-tensor trop grand")

    indices = rng.choice(
        np.arange(nweights),
        size=args.samples_per_tensor,
        replace=False
    )

    for widx in indices:
        for bit in bits:
            faults.append(
                (tensor_index, short_name, int(widx), int(bit))
            )

print("===== SWEEP F1 PRELIMINAIRE =====")
print("Modèle :", args.model)
print(f"Baseline : {100*base_acc:.2f}% ({int(correct_base.sum())}/500)")
print("Seed :", args.seed)
print("Bits :", bits)
print("Poids/tensor :", args.samples_per_tensor)
print("Nombre total de fautes :", len(faults))
print()

rows = []

for k, (tensor_index, short_name, widx, bit) in enumerate(faults, start=1):

    faulty_raw, meta = build_faulty_model(
        raw_model,
        tensor_index,
        widx,
        bit
    )

    scores_fault, preds_fault = run_model(
        faulty_raw,
        Xq
    )

    correct_fault = preds_fault == y
    fault_acc = float(np.mean(correct_fault))

    pred_changed = preds_fault != preds_base
    scores_changed = np.any(scores_fault != scores_base, axis=1)
    new_errors = (~correct_fault) & correct_base
    recovered = correct_fault & (~correct_base)

    fault_critical = np.array(
        [is_critical(y[i], preds_fault[i]) for i in range(len(y))],
        dtype=bool
    )
    new_critical = fault_critical & (~base_critical)

    score_diff = (
        scores_fault.astype(np.int16)
        - scores_base.astype(np.int16)
    )

    row = {
        "fault_id": k,
        "tensor_short": short_name,
        **meta,
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
        "faulty_model_sha256": hashlib.sha256(faulty_raw).hexdigest(),
    }

    rows.append(row)

    print(
        f"[{k:02d}/{len(faults)}] "
        f"{short_name:14s} w={widx:5d} bit={bit} | "
        f"{meta['old_value']:4d}->{meta['new_value']:4d} | "
        f"acc={100*fault_acc:6.2f}% | "
        f"Δ={100*(fault_acc-base_acc):+6.2f} | "
        f"predΔ={int(pred_changed.sum()):3d} | "
        f"newErr={int(new_errors.sum()):3d} | "
        f"newCrit={int(new_critical.sum()):2d}"
    )

with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Classements
by_acc = sorted(
    rows,
    key=lambda r: (
        r["delta_accuracy_points"],
        -r["new_critical_confusions"],
        -r["predictions_changed"]
    )
)

by_critical = sorted(
    rows,
    key=lambda r: (
        -r["new_critical_confusions"],
        r["delta_accuracy_points"]
    )
)

print()
print("===== TOP 10 - PLUS FORTE DEGRADATION D'ACCURACY =====")
for r in by_acc[:10]:
    print(
        f"{r['tensor_short']:14s} "
        f"w={r['weight_index']:5d} bit={r['bit']} | "
        f"Δacc={r['delta_accuracy_points']:+.2f} pts | "
        f"predΔ={r['predictions_changed']:3d} | "
        f"newErr={r['new_errors']:3d} | "
        f"newCrit={r['new_critical_confusions']:2d}"
    )

print()
print("===== TOP 10 - NOUVELLES CONFUSIONS CRITIQUES =====")
for r in by_critical[:10]:
    print(
        f"{r['tensor_short']:14s} "
        f"w={r['weight_index']:5d} bit={r['bit']} | "
        f"newCrit={r['new_critical_confusions']:2d} | "
        f"Δacc={r['delta_accuracy_points']:+.2f} pts | "
        f"predΔ={r['predictions_changed']:3d}"
    )

print()
print("✅ Résultats :", args.output)
print()
print("Pour matérialiser ensuite une faute intéressante :")
print(
    "python 04_f1_bitflip.py --model model_original.tflite "
    "--tensor-index <TENSOR> --weight-index <INDEX> "
    "--bit <BIT> --output faulty_selected.tflite"
)
