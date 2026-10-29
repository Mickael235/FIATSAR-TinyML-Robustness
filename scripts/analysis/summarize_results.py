from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2] / "results"

print("=== F1 ===")
f1 = pd.read_csv(root / "f1" / "comparison_runtime_vs_prefaulted.csv")
print(f1[[
    "case",
    "nano_runtime_accuracy_percent",
    "nano_runtime_delta_points",
    "runtime_prediction_changed_count"
]].to_string(index=False))

print("\n=== F2 — sensitivity by bit ===")
f2 = pd.read_csv(root / "f2" / "sensitivity_layer_bit.csv")
cols = ["layer", "bit", "prediction_changed_percent", "new_error_percent"]
print(
    f2.sort_values(["layer", "bit"])[cols].to_string(index=False)
)

print("\n=== F3 — material candidates ===")
f3 = pd.read_csv(root / "f3" / "material_candidates.csv")
print(f3.head(10).to_string(index=False))

print("\n=== F4 ===")
f4 = pd.read_csv(root / "f4" / "summary.csv")
print(f4.to_string(index=False))
