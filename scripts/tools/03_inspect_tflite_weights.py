import argparse
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb

TYPE_NAMES = {
    schema_fb.TensorType.FLOAT32: "FLOAT32",
    schema_fb.TensorType.INT32: "INT32",
    schema_fb.TensorType.UINT8: "UINT8",
    schema_fb.TensorType.INT64: "INT64",
    schema_fb.TensorType.STRING: "STRING",
    schema_fb.TensorType.BOOL: "BOOL",
    schema_fb.TensorType.INT16: "INT16",
    schema_fb.TensorType.COMPLEX64: "COMPLEX64",
    schema_fb.TensorType.INT8: "INT8",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="model_original.tflite")
args = parser.parse_args()

buf = open(args.model, "rb").read()
model = schema_fb.Model.GetRootAsModel(buf, 0)
sg = model.Subgraphs(0)

print("===== TENSEURS / BUFFERS DU MODELE =====")
print("Fichier :", args.model)
print("Tenseurs :", sg.TensorsLength())
print()

candidates = []

for i in range(sg.TensorsLength()):
    t = sg.Tensors(i)
    name = (t.Name() or b"").decode("utf-8", errors="replace")
    shape = [t.Shape(j) for j in range(t.ShapeLength())]
    ttype = TYPE_NAMES.get(t.Type(), str(t.Type()))
    bidx = t.Buffer()

    nbytes = 0
    if 0 <= bidx < model.BuffersLength():
        b = model.Buffers(bidx)
        nbytes = b.DataLength()

    print(
        f"tensor={i:2d} | buffer={bidx:2d} | "
        f"type={ttype:6s} | shape={shape!s:18s} | "
        f"bytes={nbytes:6d} | name={name}"
    )

    # Candidat F1 = constante INT8 non vide.
    if t.Type() == schema_fb.TensorType.INT8 and nbytes > 0:
        candidates.append((i, bidx, name, shape, nbytes))

print()
print("===== CANDIDATS F1 : POIDS INT8 CONSTANTS =====")
for i, bidx, name, shape, nbytes in candidates:
    print(
        f"tensor={i} | buffer={bidx} | shape={shape} | "
        f"{nbytes} poids INT8 | {nbytes*8} bits | {name}"
    )

print()
print("Attendu pour tiny_conv original :")
print("- tensor 6 : Conv2D, shape [8,10,8,1], 640 poids INT8")
print("- tensor 4 : FullyConnected, shape [12,4000], 48000 poids INT8")
