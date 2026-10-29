import argparse
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


def save_model(model_obj, path):
    builder = flatbuffers.Builder(0)
    offset = model_obj.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    raw = bytes(builder.Output())
    Path(path).write_bytes(raw)
    return raw


parser = argparse.ArgumentParser(
    description=(
        "Construit un modele TinyConv F3-Conv : "
        "seules les k premieres contributions du noyau Conv2D sont conservees."
    )
)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--k", type=int, default=79)
parser.add_argument("--output", default=None)
args = parser.parse_args()

if not (0 <= args.k <= 80):
    raise ValueError("k doit etre dans [0,80].")

if args.output is None:
    args.output = f"f3_conv_k{args.k}.tflite"

raw = Path(args.model).read_bytes()
root = schema_fb.Model.GetRootAsModel(raw, 0)
m = schema_fb.ModelT.InitFromObj(root)

if len(m.subgraphs) != 1:
    raise RuntimeError("Un seul subgraph attendu.")

sg = m.subgraphs[0]

conv_tensor_idx = None

for i, t in enumerate(sg.tensors):
    shape = tuple(int(x) for x in t.shape)
    if shape == (8, 10, 8, 1):
        conv_tensor_idx = i
        break

if conv_tensor_idx is None:
    raise RuntimeError("Poids Conv2D [8,10,8,1] introuvables.")

conv_tensor = sg.tensors[conv_tensor_idx]
buffer_idx = int(conv_tensor.buffer)
shape = tuple(int(x) for x in conv_tensor.shape)

weights = np.frombuffer(
    bytes(m.buffers[buffer_idx].data),
    dtype=np.int8
).copy().reshape(shape)

out_channels = shape[0]
terms = int(np.prod(shape[1:]))

q = conv_tensor.quantization

if q is None or q.zeroPoint is None or len(q.zeroPoint) == 0:
    zps = np.zeros(out_channels, dtype=np.int8)
else:
    raw_zps = np.asarray(q.zeroPoint, dtype=np.int64)
    if len(raw_zps) == 1:
        zps = np.full(out_channels, int(raw_zps[0]), dtype=np.int8)
    elif len(raw_zps) == out_channels:
        zps = raw_zps.astype(np.int8)
    else:
        raise RuntimeError(f"Zero-points inattendus : {raw_zps}")

w2 = weights.reshape(out_channels, terms)

changed = []

if args.k < terms:
    for oc in range(out_channels):
        for j in range(args.k, terms):
            old = int(w2[oc, j])
            new = int(zps[oc])
            if old != new:
                changed.append((oc, j, old, new))
            w2[oc, j] = np.int8(new)

m.buffers[buffer_idx].data = bytearray(
    w2.reshape(shape).astype(np.int8).tobytes()
)

new_raw = save_model(m, args.output)

print("========================================")
print(" F3 CONV - MODELE TRONQUE")
print("========================================")
print("Tensor poids Conv :", conv_tensor_idx)
print("Shape             :", shape)
print("Contributions     :", terms)
print("k conserve        :", args.k)
print("Contributions skip:", terms - args.k)
print("Poids modifies    :", len(changed))
print("Fichier           :", args.output)
print("Taille            :", len(new_raw), "octets")
print()

# ------------------------------------------------------------
# Validation PC sur les 500 entrees
# ------------------------------------------------------------

base = np.load(args.baseline)
Xq = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)
base_scores = base["scores_int8"].astype(np.int8)
base_preds = base["preds"].astype(np.int64)

interp = tf.lite.Interpreter(
    model_path=args.output,
    num_threads=1
)
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]

fault_scores = np.empty((len(Xq), 12), dtype=np.int8)
fault_preds = np.empty(len(Xq), dtype=np.int64)

for i in range(len(Xq)):
    interp.set_tensor(inp["index"], Xq[i:i+1])
    interp.invoke()
    s = interp.get_tensor(out["index"])[0]
    fault_scores[i] = s
    fault_preds[i] = int(np.argmax(s))

base_correct = base_preds == y
fault_correct = fault_preds == y

base_acc = 100.0 * np.mean(base_correct)
fault_acc = 100.0 * np.mean(fault_correct)

new_error = base_correct & (~fault_correct)
recovered = (~base_correct) & fault_correct

base_crit = np.array(
    [critical(t, p) for t, p in zip(y, base_preds)],
    dtype=bool
)
fault_crit = np.array(
    [critical(t, p) for t, p in zip(y, fault_preds)],
    dtype=bool
)
new_crit = (~base_crit) & fault_crit

print("===== VALIDATION PC =====")
print(f"Nominal accuracy       : {base_acc:.2f}%")
print(f"F3 Conv k={args.k} accuracy : {fault_acc:.2f}%")
print(f"Delta                  : {fault_acc-base_acc:+.2f} pts")
print(f"Predictions changees   : {int(np.sum(base_preds != fault_preds))}")
print(f"Nouvelles erreurs      : {int(np.sum(new_error))}")
print(f"Erreurs recuperees     : {int(np.sum(recovered))}")
print(f"Nouvelles critiques    : {int(np.sum(new_crit))}")
print()

if args.k == 80:
    exact = int(np.sum(np.all(fault_scores == base_scores, axis=1)))
    print(f"k=80 bit-exact baseline : {exact}/{len(Xq)}")
    if exact != len(Xq):
        raise RuntimeError("k=80 ne reproduit pas le nominal.")

print("✅ MODELE F3 CONV PRET")
