import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Convertit un .tflite en header C avec symbole personnalisable."
)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--symbol", required=True)
args = parser.parse_args()

symbol = re.sub(r"[^A-Za-z0-9_]", "_", args.symbol)
if not symbol or symbol[0].isdigit():
    raise ValueError("Nom de symbole C invalide.")

data = Path(args.model).read_bytes()
guard = re.sub(r"[^A-Za-z0-9_]", "_", Path(args.output).name.upper()) + "_"

lines = [
    f"#ifndef {guard}",
    f"#define {guard}",
    "",
    "#include <stdint.h>",
    "",
    f"alignas(16) const unsigned char {symbol}[] = {{",
]

for i in range(0, len(data), 12):
    chunk = data[i:i+12]
    lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")

lines += [
    "};",
    f"const int {symbol}_len = {len(data)};",
    "",
    f"#endif  // {guard}",
    "",
]

Path(args.output).write_text("\n".join(lines), encoding="utf-8")

print("✅", args.output)
print("Source  :", args.model)
print("Taille  :", len(data), "octets")
print("Symbole :", symbol)
print("Longueur:", f"{symbol}_len")
