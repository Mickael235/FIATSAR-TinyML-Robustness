import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Chemin vers data/propre/tiny_conv/testing.npz")
parser.add_argument("--output", default="testset_500.npz")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

data = np.load(args.input)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)

if X.ndim != 2 or X.shape[1] != 1960:
    raise ValueError(f"X attendu [N,1960], reçu {X.shape}")
if y.shape != (len(X),):
    raise ValueError(f"y attendu ({len(X)},), reçu {y.shape}")

rng = np.random.default_rng(args.seed)
selected = []

# 42 exemples pour classes 0..7, 41 pour classes 8..11 = 500.
for cls in range(12):
    cls_idx = np.flatnonzero(y == cls)
    n = 42 if cls < 8 else 41
    if len(cls_idx) < n:
        raise RuntimeError(
            f"Classe {cls}: {len(cls_idx)} exemples seulement, {n} requis."
        )
    chosen = rng.choice(cls_idx, size=n, replace=False)
    selected.extend(chosen.tolist())

selected = np.asarray(selected, dtype=np.int64)
rng.shuffle(selected)

np.savez_compressed(
    args.output,
    X=X[selected],
    y=y[selected],
    indices=selected,
    seed=np.array(args.seed, dtype=np.int64),
)

print("✅", args.output, "créé")
print("X :", X[selected].shape, X[selected].dtype)
print("y :", y[selected].shape, y[selected].dtype)
print("seed :", args.seed)
print("Répartition :")
for cls in range(12):
    print(f"  classe {cls:2d} :", int(np.sum(y[selected] == cls)))
