import argparse
from pathlib import Path

import numpy as np
from tensorflow.lite.python import schema_py_generated as schema_fb


KNOWN_FAULTS = [
    ("Conv2D w5 b7", 6, 5, 7),
    ("Conv2D w31 b7", 6, 31, 7),
    ("FC w45024 b7", 4, 45024, 7),
]


def int8_flip(value, bit):
    u = int(np.uint8(np.int8(value)))
    u ^= (1 << bit)
    return int(np.array([u], dtype=np.uint8).view(np.int8)[0])


def get_buffer_data_start(buf):
    o = buf._tab.Offset(4)
    if o == 0:
        return None
    return int(buf._tab.Vector(o))


def tensor_name(tensor):
    try:
        return tensor.Name().decode("utf-8", errors="replace")
    except Exception:
        return ""


def tensor_shape(tensor):
    return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trouve les offsets absolus des poids INT8 dans un fichier .tflite "
            "et genere f1_offsets.h pour l'injection F1 runtime."
        )
    )
    parser.add_argument("--model", default="model_original.tflite")
    parser.add_argument("--output", default="f1_offsets.h")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    raw = model_path.read_bytes()
    model = schema_fb.Model.GetRootAsModel(raw, 0)

    if model.SubgraphsLength() != 1:
        raise RuntimeError(
            f"Un seul subgraph attendu, trouve: {model.SubgraphsLength()}"
        )

    sg = model.Subgraphs(0)
    rows = []

    for tensor_index in range(sg.TensorsLength()):
        tensor = sg.Tensors(tensor_index)

        if tensor.Type() != schema_fb.TensorType.INT8:
            continue

        buffer_index = int(tensor.Buffer())
        if buffer_index <= 0 or buffer_index >= model.BuffersLength():
            continue

        buf = model.Buffers(buffer_index)
        data_len = int(buf.DataLength())

        if data_len <= 0:
            continue

        data_start = get_buffer_data_start(buf)
        if data_start is None:
            continue

        data_np = buf.DataAsNumpy()
        if isinstance(data_np, int):
            continue

        data_bytes = bytes(data_np)

        if raw[data_start:data_start + data_len] != data_bytes:
            raise RuntimeError(
                f"Offset FlatBuffer incoherent pour tensor {tensor_index}"
            )

        shape = tensor_shape(tensor)
        expected_elems = int(np.prod(shape)) if shape else 0

        if expected_elems != data_len:
            continue

        rows.append({
            "tensor_index": tensor_index,
            "buffer_index": buffer_index,
            "name": tensor_name(tensor),
            "shape": shape,
            "data_offset": data_start,
            "byte_count": data_len,
        })

    if not rows:
        raise RuntimeError("Aucun tensor constant INT8 trouve.")

    print("============================================================")
    print(" F1 OFFSETS - TFLITE")
    print("============================================================")
    print(f"Modele : {model_path}")
    print(f"Taille : {len(raw)} octets")
    print()

    for r in rows:
        print(
            f"tensor {r['tensor_index']:2d} | "
            f"buffer {r['buffer_index']:2d} | "
            f"shape={r['shape']} | "
            f"bytes={r['byte_count']:6d} | "
            f"offset={r['data_offset']:6d} | "
            f"{r['name']}"
        )

    by_tensor = {r["tensor_index"]: r for r in rows}

    print()
    print("===== VERIFICATION DES 3 FAUTES CONNUES =====")

    known_lines = []

    for label, tensor_index, weight_index, bit in KNOWN_FAULTS:
        if tensor_index not in by_tensor:
            print(f"[ABSENT] {label}: tensor {tensor_index} non trouve")
            continue

        r = by_tensor[tensor_index]

        if not (0 <= weight_index < r["byte_count"]):
            print(
                f"[ERREUR] {label}: index {weight_index} hors plage "
                f"0..{r['byte_count'] - 1}"
            )
            continue

        absolute = r["data_offset"] + weight_index
        old_u8 = raw[absolute]
        old_i8 = int(np.array([old_u8], dtype=np.uint8).view(np.int8)[0])
        new_i8 = int8_flip(old_i8, bit)

        print(
            f"{label:18s} | tensor={tensor_index} | index={weight_index:5d} | "
            f"bit={bit} | offset absolu={absolute:6d} | "
            f"{old_i8:4d} -> {new_i8:4d}"
        )

        known_lines.append(
            (label, tensor_index, weight_index, bit, absolute, old_i8, new_i8)
        )

    header = []
    header.append("#pragma once")
    header.append("")
    header.append("// ============================================================")
    header.append("// AUTO-GENERE PAR f1_offsets.py")
    header.append(f"// Modele : {model_path.name}")
    header.append(f"// Taille : {len(raw)} octets")
    header.append("// Convention des index : ordre aplati DU FICHIER TFLITE.")
    header.append("// Ne pas confondre avec l'ordre Keras / SavedModel.")
    header.append("// ============================================================")
    header.append("")
    header.append("#include <stddef.h>")
    header.append("#include <stdint.h>")
    header.append("")
    header.append(f"constexpr size_t F1_MODEL_SIZE = {len(raw)};")
    header.append("")
    header.append("struct F1TensorOffsetInfo {")
    header.append("  int tensor_index;")
    header.append("  size_t data_offset;")
    header.append("  size_t byte_count;")
    header.append("};")
    header.append("")
    header.append("constexpr F1TensorOffsetInfo F1_WEIGHT_TENSORS[] = {")

    for r in rows:
        header.append(
            f"  {{{r['tensor_index']}, {r['data_offset']}, {r['byte_count']}}}, "
            f"// buffer {r['buffer_index']} {r['name']} shape={r['shape']}"
        )

    header.append("};")
    header.append("")
    header.append(
        "constexpr size_t F1_WEIGHT_TENSOR_COUNT = "
        "sizeof(F1_WEIGHT_TENSORS) / sizeof(F1_WEIGHT_TENSORS[0]);"
    )
    header.append("")
    header.append("// Fautes F1 deja validees lors de la campagne TinyConv :")

    for label, tensor_index, weight_index, bit, absolute, old_i8, new_i8 in known_lines:
        header.append(
            f"// {label}: tensor {tensor_index}, index {weight_index}, bit {bit}, "
            f"offset {absolute}, {old_i8} -> {new_i8}"
        )

    Path(args.output).write_text("\n".join(header) + "\n", encoding="utf-8")

    print()
    print(f"✅ Header genere : {args.output}")
    print()
    print("Etape suivante :")
    print(
        "  Copier f1_offsets.h dans le dossier du firmware, puis integrer "
        "f1_runtime_snippet.cpp."
    )


if __name__ == "__main__":
    main()
