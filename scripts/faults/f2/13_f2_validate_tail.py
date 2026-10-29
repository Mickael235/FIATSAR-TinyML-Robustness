import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def qinfo(detail):
    q = detail.get("quantization", (0.0, 0))
    return f"scale={q[0]} zero_point={q[1]}"


parser = argparse.ArgumentParser(
    description="Valide les sous-modeles F2 en comparant leurs sorties au TinyConv complet."
)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--fc-tail", default="f2_tail_fc_softmax.tflite")
parser.add_argument("--softmax-tail", default="f2_tail_softmax.tflite")
parser.add_argument("--n", type=int, default=500)
args = parser.parse_args()

for p in [args.model, args.baseline, args.fc_tail, args.softmax_tail]:
    if not Path(p).exists():
        raise FileNotFoundError(p)

data = np.load(args.baseline)
Xq = data["X_int8"].astype(np.int8)
scores_ref_saved = data["scores_int8"].astype(np.int8)

n = min(args.n, len(Xq))

# Preserve les tenseurs intermédiaires.
full = tf.lite.Interpreter(
    model_path=args.model,
    experimental_preserve_all_tensors=True,
    num_threads=1,
)
full.allocate_tensors()

tail_fc = tf.lite.Interpreter(
    model_path=args.fc_tail,
    num_threads=1,
)
tail_fc.allocate_tensors()

tail_sm = tf.lite.Interpreter(
    model_path=args.softmax_tail,
    num_threads=1,
)
tail_sm.allocate_tensors()

full_in = full.get_input_details()[0]
full_out = full.get_output_details()[0]

fc_in = tail_fc.get_input_details()[0]
fc_out = tail_fc.get_output_details()[0]

sm_in = tail_sm.get_input_details()[0]
sm_out = tail_sm.get_output_details()[0]

details = {d["index"]: d for d in full.get_tensor_details()}

for idx in [9, 10, 11]:
    if idx not in details:
        raise RuntimeError(f"Tensor {idx} introuvable dans le modele complet.")

print("===== TENSEURS F2 =====")
for idx in [9, 10, 11]:
    d = details[idx]
    print(
        f"tensor {idx:2d} | shape={list(d['shape'])} | "
        f"dtype={d['dtype']} | {qinfo(d)} | name={d['name']}"
    )

print()
print("===== ENTREES DES TAILS =====")
print(
    "FC tail      :",
    list(fc_in["shape"]),
    fc_in["dtype"],
    qinfo(fc_in),
)
print(
    "Softmax tail :",
    list(sm_in["shape"]),
    sm_in["dtype"],
    qinfo(sm_in),
)

exact_fc = 0
exact_sm = 0
exact_saved = 0
pred_fc = 0
pred_sm = 0

max_diff_fc = 0
max_diff_sm = 0

for i in range(n):
    full.set_tensor(full_in["index"], Xq[i:i+1])
    full.invoke()

    activation_fc = full.get_tensor(9).copy()
    logits = full.get_tensor(10).copy()
    out_full = full.get_tensor(full_out["index"]).copy()

    # Vérification baseline sauvegardée.
    if np.array_equal(out_full[0], scores_ref_saved[i]):
        exact_saved += 1

    # Tail FC + Softmax.
    tail_fc.set_tensor(fc_in["index"], activation_fc)
    tail_fc.invoke()
    out_fc = tail_fc.get_tensor(fc_out["index"]).copy()

    # Tail Softmax.
    tail_sm.set_tensor(sm_in["index"], logits)
    tail_sm.invoke()
    out_sm = tail_sm.get_tensor(sm_out["index"]).copy()

    if np.array_equal(out_full, out_fc):
        exact_fc += 1
    if np.array_equal(out_full, out_sm):
        exact_sm += 1

    if int(np.argmax(out_full[0])) == int(np.argmax(out_fc[0])):
        pred_fc += 1
    if int(np.argmax(out_full[0])) == int(np.argmax(out_sm[0])):
        pred_sm += 1

    max_diff_fc = max(
        max_diff_fc,
        int(np.max(np.abs(
            out_full.astype(np.int16) - out_fc.astype(np.int16)
        )))
    )
    max_diff_sm = max(
        max_diff_sm,
        int(np.max(np.abs(
            out_full.astype(np.int16) - out_sm.astype(np.int16)
        )))
    )

print()
print("===== VALIDATION TAILS F2 =====")
print(f"Echantillons                  : {n}")
print(f"Modele complet = baseline NPZ : {exact_saved}/{n}")
print()
print(
    f"Tensor 9 -> FC + Softmax      : "
    f"{exact_fc}/{n} scores bit-exacts"
)
print(
    f"Accord prediction             : "
    f"{pred_fc}/{n}"
)
print(
    f"Ecart score max               : "
    f"{max_diff_fc}"
)
print()
print(
    f"Tensor 10 -> Softmax          : "
    f"{exact_sm}/{n} scores bit-exacts"
)
print(
    f"Accord prediction             : "
    f"{pred_sm}/{n}"
)
print(
    f"Ecart score max               : "
    f"{max_diff_sm}"
)

if exact_fc == n and exact_sm == n:
    print()
    print("✅ VALIDATION F2 OK")
    print("On peut injecter des corruptions INT8 dans les tensors 9 et 10.")
else:
    print()
    print("❌ VALIDATION F2 NON BIT-EXACTE")
    print("Ne pas lancer la campagne F2 avant correction.")
