import argparse
from pathlib import Path

import flatbuffers
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb


def save_model(model_obj, path):
    builder = flatbuffers.Builder(0)
    offset = model_obj.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    raw = bytes(builder.Output())
    Path(path).write_bytes(raw)
    return raw


parser = argparse.ArgumentParser(
    description="Expose les logits (tensor 10) comme 2e sortie TFLite pour compatibilite TFLM Arduino."
)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--output", default="f2_model_original_with_logits.tflite")
args = parser.parse_args()

raw = Path(args.model).read_bytes()
root = schema_fb.Model.GetRootAsModel(raw, 0)
m = schema_fb.ModelT.InitFromObj(root)

if len(m.subgraphs) != 1:
    raise RuntimeError("Un seul subgraph attendu.")

sg = m.subgraphs[0]

# TinyConv :
# tensor 10 = logits FullyConnected
# tensor 11 = Softmax final
LOGITS_TENSOR = 10
SOFTMAX_TENSOR = 11

print("Sorties originales :", list(sg.outputs))

# output(0) reste le Softmax nominal ; output(1) expose les logits.
sg.outputs = [SOFTMAX_TENSOR, LOGITS_TENSOR]

new_raw = save_model(m, args.output)

print("Nouvelles sorties  :", list(sg.outputs))
print("Fichier             :", args.output)
print("Taille              :", len(new_raw), "octets")

# ------------------------------------------------------------
# Validation PC sur les 500 entrees
# ------------------------------------------------------------

base = np.load(args.baseline)
Xq = base["X_int8"].astype(np.int8)
scores_saved = base["scores_int8"].astype(np.int8)

interp = tf.lite.Interpreter(model_path=args.output, num_threads=1)
interp.allocate_tensors()

inp = interp.get_input_details()[0]
outs = interp.get_output_details()

print()
print("===== SORTIES DU MODELE INSTRUMENTE =====")
for i, d in enumerate(outs):
    print(
        f"output({i}) | tensor={d['index']} | "
        f"shape={list(d['shape'])} | dtype={d['dtype']} | "
        f"quant={d['quantization']} | name={d['name']}"
    )

if len(outs) != 2:
    raise RuntimeError("Le modele doit avoir exactement 2 sorties.")

exact = 0

for i in range(len(Xq)):
    interp.set_tensor(inp["index"], Xq[i:i+1])
    interp.invoke()

    softmax = interp.get_tensor(outs[0]["index"])[0]

    if np.array_equal(softmax, scores_saved[i]):
        exact += 1

print()
print("===== VALIDATION =====")
print(f"Softmax original preserve : {exact}/{len(Xq)} bit-exact")

if exact != len(Xq):
    raise RuntimeError("Le modele instrumente modifie la sortie nominale.")

print("✅ MODELE F2 COMPATIBLE VALIDE")
print("Le tensor logits est maintenant accessible via output(1) sur TFLM.")
