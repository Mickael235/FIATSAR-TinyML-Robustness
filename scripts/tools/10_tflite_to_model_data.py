import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Convertit un .tflite en model_data.h compatible avec le firmware UART Nano."
)
parser.add_argument("--model", required=True)
parser.add_argument("--output", default="model_data.h")
args = parser.parse_args()

data = Path(args.model).read_bytes()

lines = []
lines.append("#ifndef MODEL_DATA_H_")
lines.append("#define MODEL_DATA_H_")
lines.append("")
lines.append("#include <stdint.h>")
lines.append("")
lines.append("alignas(16) const unsigned char g_model[] = {")

for i in range(0, len(data), 12):
    chunk = data[i:i+12]
    lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")

lines.append("};")
lines.append(f"const int g_model_len = {len(data)};")
lines.append("")
lines.append("#endif  // MODEL_DATA_H_")
lines.append("")

Path(args.output).write_text("\n".join(lines), encoding="utf-8")

print("✅", args.output)
print("Modèle source :", args.model)
print("Taille        :", len(data), "octets")
print("Symboles      : g_model / g_model_len")
