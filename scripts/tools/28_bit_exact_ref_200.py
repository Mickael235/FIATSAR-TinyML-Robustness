import argparse
import os
from pathlib import Path

# A placer AVANT TensorFlow pour rendre le test PC aussi stable que possible.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf


def build_interpreter(model_path, mode):
    kwargs = {"model_path": str(model_path)}

    if mode == "normal":
        return tf.lite.Interpreter(**kwargs)

    if mode == "ref":
        try:
            resolver = tf.lite.experimental.OpResolverType.BUILTIN_REF
        except AttributeError as e:
            raise RuntimeError(
                "Cette version de TensorFlow n'expose pas "
                "tf.lite.experimental.OpResolverType.BUILTIN_REF"
            ) from e

        return tf.lite.Interpreter(
            **kwargs,
            experimental_op_resolver_type=resolver
        )

    raise ValueError(mode)


def run_model(model_path, X, mode):
    itp = build_interpreter(model_path, mode)
    itp.allocate_tensors()

    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]

    if inp["dtype"] != np.int8:
        raise RuntimeError(f"Entree attendue int8, trouvee {inp['dtype']}")
    if out["dtype"] != np.int8:
        raise RuntimeError(f"Sortie attendue int8, trouvee {out['dtype']}")

    outputs = np.empty((len(X), int(np.prod(out["shape"]))), dtype=np.int8)

    for i, x in enumerate(X):
        x = np.asarray(x, dtype=np.int8).reshape(inp["shape"])
        itp.set_tensor(inp["index"], x)
        itp.invoke()
        y = itp.get_tensor(out["index"]).astype(np.int8).reshape(-1)
        outputs[i] = y

    return outputs, inp, out


def load_nano_scores(csv_path, n):
    df = pd.read_csv(csv_path)

    cols = [f"score_{i}" for i in range(12)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Colonnes scores absentes du CSV Nano : " + ", ".join(missing)
        )

    if len(df) < n:
        raise RuntimeError(
            f"CSV Nano contient {len(df)} lignes, mais {n} demandees."
        )

    scores = df.loc[:n-1, cols].to_numpy(dtype=np.int8)

    if "prediction_id" in df.columns:
        pred = df.loc[:n-1, "prediction_id"].to_numpy(dtype=np.int64)
    else:
        pred = np.argmax(scores.astype(np.int16), axis=1)

    return scores, pred, df.iloc[:n].copy()


def exact_rows(a, b):
    return np.all(a == b, axis=1)


def byte_match_count(a, b):
    return int(np.sum(a == b))


def prediction(x):
    return np.argmax(x.astype(np.int16), axis=1)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Diagnostic bit-exact PC TFLite normal vs BUILTIN_REF vs Nano "
            "sur un lot fixe."
        )
    )
    p.add_argument("--model", default="model_original.tflite")
    p.add_argument("--baseline", default="baseline_500.npz")
    p.add_argument(
        "--nano-csv",
        default=r"resultats_runtime\f1_runtime_nominal_500.csv"
    )
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--output", default="bit_exact_ref_200.csv")
    p.add_argument("--summary", default="bit_exact_ref_resume.txt")
    args = p.parse_args()

    model_path = Path(args.model)
    baseline_path = Path(args.baseline)
    nano_csv = Path(args.nano_csv)

    base = np.load(baseline_path)

    if "X_int8" not in base.files:
        raise RuntimeError(
            f"Cle X_int8 absente de {baseline_path}. Cles: {base.files}"
        )

    X = base["X_int8"].astype(np.int8)

    n = min(args.samples, len(X))
    X = X[:n]

    nano_scores, nano_pred, nano_df = load_nano_scores(nano_csv, n)

    print("============================================================")
    print(" BIT-EXACT PC / NANO - TEST BUILTIN_REF")
    print("============================================================")
    print(f"Modele     : {model_path}")
    print(f"Baseline   : {baseline_path}")
    print(f"Nano CSV   : {nano_csv}")
    print(f"Echantillons : {n}")
    print()

    print("[1/2] Inference PC TFLite normale...")
    normal_scores, inp_normal, out_normal = run_model(
        model_path, X, "normal"
    )

    print("[2/2] Inference PC TFLite BUILTIN_REF...")
    ref_scores, inp_ref, out_ref = run_model(
        model_path, X, "ref"
    )

    normal_pred = prediction(normal_scores)
    ref_pred = prediction(ref_scores)

    exact_normal_nano = exact_rows(normal_scores, nano_scores)
    exact_ref_nano = exact_rows(ref_scores, nano_scores)
    exact_normal_ref = exact_rows(normal_scores, ref_scores)

    total_bytes = n * 12

    normal_nano_bytes = byte_match_count(normal_scores, nano_scores)
    ref_nano_bytes = byte_match_count(ref_scores, nano_scores)
    normal_ref_bytes = byte_match_count(normal_scores, ref_scores)

    pred_normal_nano = int(np.sum(normal_pred == nano_pred))
    pred_ref_nano = int(np.sum(ref_pred == nano_pred))
    pred_normal_ref = int(np.sum(normal_pred == ref_pred))

    rows = []

    for i in range(n):
        row = {
            "sample": i,
            "nano_pred": int(nano_pred[i]),
            "pc_normal_pred": int(normal_pred[i]),
            "pc_ref_pred": int(ref_pred[i]),
            "scores_exact_pc_normal_vs_nano": bool(exact_normal_nano[i]),
            "scores_exact_pc_ref_vs_nano": bool(exact_ref_nano[i]),
            "scores_exact_pc_normal_vs_ref": bool(exact_normal_ref[i]),
            "matching_bytes_pc_normal_vs_nano": int(
                np.sum(normal_scores[i] == nano_scores[i])
            ),
            "matching_bytes_pc_ref_vs_nano": int(
                np.sum(ref_scores[i] == nano_scores[i])
            ),
        }

        for j in range(12):
            row[f"nano_score_{j}"] = int(nano_scores[i, j])
            row[f"pc_normal_score_{j}"] = int(normal_scores[i, j])
            row[f"pc_ref_score_{j}"] = int(ref_scores[i, j])

        rows.append(row)

    pd.DataFrame(rows).to_csv(args.output, index=False)

    lines = []
    lines.append("BIT-EXACT PC / NANO - BUILTIN_REF")
    lines.append("=" * 50)
    lines.append(f"Echantillons : {n}")
    lines.append("")
    lines.append(
        f"PC normal vs Nano - vecteurs exacts : "
        f"{int(exact_normal_nano.sum())}/{n} "
        f"({100*exact_normal_nano.mean():.2f} %)"
    )
    lines.append(
        f"PC REF vs Nano    - vecteurs exacts : "
        f"{int(exact_ref_nano.sum())}/{n} "
        f"({100*exact_ref_nano.mean():.2f} %)"
    )
    lines.append(
        f"PC normal vs REF  - vecteurs exacts : "
        f"{int(exact_normal_ref.sum())}/{n} "
        f"({100*exact_normal_ref.mean():.2f} %)"
    )
    lines.append("")
    lines.append(
        f"PC normal vs Nano - octets egaux : "
        f"{normal_nano_bytes}/{total_bytes} "
        f"({100*normal_nano_bytes/total_bytes:.2f} %)"
    )
    lines.append(
        f"PC REF vs Nano    - octets egaux : "
        f"{ref_nano_bytes}/{total_bytes} "
        f"({100*ref_nano_bytes/total_bytes:.2f} %)"
    )
    lines.append(
        f"PC normal vs REF  - octets egaux : "
        f"{normal_ref_bytes}/{total_bytes} "
        f"({100*normal_ref_bytes/total_bytes:.2f} %)"
    )
    lines.append("")
    lines.append(
        f"Predictions PC normal vs Nano : {pred_normal_nano}/{n}"
    )
    lines.append(
        f"Predictions PC REF vs Nano    : {pred_ref_nano}/{n}"
    )
    lines.append(
        f"Predictions PC normal vs REF  : {pred_normal_ref}/{n}"
    )
    lines.append("")
    lines.append(
        f"Input quant PC normal : scale={inp_normal['quantization'][0]} "
        f"zp={inp_normal['quantization'][1]}"
    )
    lines.append(
        f"Input quant PC REF    : scale={inp_ref['quantization'][0]} "
        f"zp={inp_ref['quantization'][1]}"
    )
    lines.append(
        f"Output quant PC normal: scale={out_normal['quantization'][0]} "
        f"zp={out_normal['quantization'][1]}"
    )
    lines.append(
        f"Output quant PC REF   : scale={out_ref['quantization'][0]} "
        f"zp={out_ref['quantization'][1]}"
    )
    lines.append("")

    if int(exact_ref_nano.sum()) == n:
        conclusion = (
            "CONCLUSION : 200/200 exacts avec BUILTIN_REF. "
            "La divergence venait du backend PC optimise."
        )
    elif np.array_equal(normal_scores, ref_scores):
        conclusion = (
            "CONCLUSION : BUILTIN_REF produit exactement les memes sorties "
            "que le TFLite PC normal, mais celles-ci divergent encore de la Nano. "
            "La piste backend PC/XNNPACK n'explique donc pas la non-conformite."
        )
    else:
        conclusion = (
            "CONCLUSION : BUILTIN_REF modifie les sorties PC mais ne donne pas "
            "un accord complet avec la Nano. Il faut poursuivre cote TFLM/Nano, "
            "en particulier la couche FullyConnected."
        )

    lines.append(conclusion)

    summary = "\n".join(lines) + "\n"
    Path(args.summary).write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"CSV detail : {args.output}")
    print(f"Resume     : {args.summary}")


if __name__ == "__main__":
    main()
