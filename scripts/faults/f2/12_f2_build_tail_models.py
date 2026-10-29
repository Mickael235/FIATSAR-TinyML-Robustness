import argparse
import hashlib
from pathlib import Path

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_model(model_obj, path):
    builder = flatbuffers.Builder(0)
    offset = model_obj.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    raw = bytes(builder.Output())
    Path(path).write_bytes(raw)
    return raw


parser = argparse.ArgumentParser(
    description="Construit deux sous-modeles TFLite pour injecter F2 dans les activations TinyConv."
)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--fc-tail", default="f2_tail_fc_softmax.tflite")
parser.add_argument("--softmax-tail", default="f2_tail_softmax.tflite")
args = parser.parse_args()

raw = Path(args.model).read_bytes()
model = schema_fb.Model.GetRootAsModel(raw, 0)
base = schema_fb.ModelT.InitFromObj(model)

if len(base.subgraphs) != 1:
    raise RuntimeError("Ce script attend un seul subgraph.")

sg = base.subgraphs[0]

print("===== OPERATEURS DU MODELE =====")
for i, op in enumerate(sg.operators):
    print(
        f"op={i:2d} | inputs={list(op.inputs)} | outputs={list(op.outputs)}"
    )

# TinyConv inspecte précédemment :
# tensor 9  = activation aplatie après Conv2D/Reshape, [1,4000] INT8
# tensor 10 = logits FullyConnected, [1,12] INT8
# tensor 11 = sortie Softmax, [1,12] INT8
TENSOR_FC_INPUT = 9
TENSOR_LOGITS = 10
TENSOR_OUTPUT = 11

fc_op_index = None
softmax_op_index = None

for i, op in enumerate(sg.operators):
    outs = list(op.outputs)
    if TENSOR_LOGITS in outs:
        fc_op_index = i
    if TENSOR_OUTPUT in outs:
        softmax_op_index = i

if fc_op_index is None:
    raise RuntimeError("Operateur produisant le tensor 10 introuvable.")
if softmax_op_index is None:
    raise RuntimeError("Operateur produisant le tensor 11 introuvable.")

print()
print("Operateur FullyConnected detecte :", fc_op_index)
print("Operateur Softmax detecte        :", softmax_op_index)

# ------------------------------------------------------------
# Tail 1 : tensor 9 -> FullyConnected -> Softmax -> tensor 11
# ------------------------------------------------------------
m1 = schema_fb.ModelT.InitFromObj(model)
sg1 = m1.subgraphs[0]
sg1.inputs = [TENSOR_FC_INPUT]
sg1.outputs = [TENSOR_OUTPUT]
sg1.operators = [
    sg1.operators[fc_op_index],
    sg1.operators[softmax_op_index],
]

raw1 = save_model(m1, args.fc_tail)

# ------------------------------------------------------------
# Tail 2 : tensor 10 -> Softmax -> tensor 11
# ------------------------------------------------------------
m2 = schema_fb.ModelT.InitFromObj(model)
sg2 = m2.subgraphs[0]
sg2.inputs = [TENSOR_LOGITS]
sg2.outputs = [TENSOR_OUTPUT]
sg2.operators = [
    sg2.operators[softmax_op_index],
]

raw2 = save_model(m2, args.softmax_tail)

print()
print("===== SOUS-MODELES F2 CREES =====")
print("Source :", args.model)
print("SHA256 :", sha256(raw))
print()
print("Tail FC + Softmax")
print("  entree tensor :", TENSOR_FC_INPUT)
print("  sortie tensor :", TENSOR_OUTPUT)
print("  fichier       :", args.fc_tail)
print("  taille        :", len(raw1), "octets")
print("  SHA256        :", sha256(raw1))
print()
print("Tail Softmax")
print("  entree tensor :", TENSOR_LOGITS)
print("  sortie tensor :", TENSOR_OUTPUT)
print("  fichier       :", args.softmax_tail)
print("  taille        :", len(raw2), "octets")
print("  SHA256        :", sha256(raw2))
print()
print("Etape suivante : valider que les deux tails reproduisent exactement le modele nominal.")
