import argparse
import hashlib
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb

def signed_int8(u):
    return u - 256 if u >= 128 else u

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tensor-index", type=int, required=True)
parser.add_argument("--weight-index", type=int, required=True)
parser.add_argument("--bit", type=int, choices=range(8), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

src = Path(args.model)
dst = Path(args.output)

raw = src.read_bytes()
model = schema_fb.Model.GetRootAsModel(raw, 0)
model_obj = schema_fb.ModelT.InitFromObj(model)

if len(model_obj.subgraphs) != 1:
    raise RuntimeError("Ce script suppose un seul subgraph.")

sg = model_obj.subgraphs[0]

if not (0 <= args.tensor_index < len(sg.tensors)):
    raise IndexError("tensor-index hors limites")

tensor = sg.tensors[args.tensor_index]

if tensor.type != schema_fb.TensorType.INT8:
    raise TypeError(
        f"Tensor {args.tensor_index} n'est pas INT8 (type={tensor.type})"
    )

buffer_index = tensor.buffer

if not (0 <= buffer_index < len(model_obj.buffers)):
    raise IndexError("Buffer du tenseur invalide")

buffer_obj = model_obj.buffers[buffer_index]

if buffer_obj.data is None:
    raise RuntimeError("Buffer vide")

# Normaliser en bytearray modifiable.
data = bytearray(buffer_obj.data)

if not (0 <= args.weight_index < len(data)):
    raise IndexError(
        f"weight-index={args.weight_index} hors limites [0,{len(data)-1}]"
    )

old_u8 = data[args.weight_index]
new_u8 = old_u8 ^ (1 << args.bit)

data[args.weight_index] = new_u8
buffer_obj.data = data

# Repacker le FlatBuffer TFLite.
builder = flatbuffers.Builder(0)
model_offset = model_obj.Pack(builder)
builder.Finish(model_offset, file_identifier=b"TFL3")
new_raw = bytes(builder.Output())

dst.write_bytes(new_raw)

print("===== F1 BIT-FLIP CREE =====")
print("Modele source :", src)
print("SHA256 source :", hashlib.sha256(raw).hexdigest())
print()
print("Tensor index  :", args.tensor_index)
name = tensor.name.decode("utf-8", errors="replace") if isinstance(tensor.name, (bytes, bytearray)) else str(tensor.name)
print("Tensor name   :", name)
print("Shape         :", list(tensor.shape))
print("Buffer index  :", buffer_index)
print("Weight index  :", args.weight_index)
print("Bit flippe    :", args.bit)
print(
    f"Valeur avant  : {signed_int8(old_u8)} "
    f"(0x{old_u8:02X}, {old_u8:08b})"
)
print(
    f"Valeur apres  : {signed_int8(new_u8)} "
    f"(0x{new_u8:02X}, {new_u8:08b})"
)
print()
print("Sortie        :", dst)
print("Taille        :", len(new_raw), "octets")
print("SHA256 sortie :", hashlib.sha256(new_raw).hexdigest())

# Vérification légère avec TFLite si TensorFlow est disponible.
try:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(dst))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    print()
    print("Verification TFLite : OK")
    print("Input  :", inp["shape"], inp["dtype"], inp["quantization"])
    print("Output :", out["shape"], out["dtype"], out["quantization"])
except Exception as e:
    print()
    print("ATTENTION : le fichier a ete cree, mais la verification TFLite a echoue :")
    print(repr(e))
