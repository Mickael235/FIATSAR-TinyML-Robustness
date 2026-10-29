import argparse
import csv
import time
from pathlib import Path

import numpy as np
import serial


LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

FAULTS = [
    {
        "name": "conv_w5_b7",
        "tensor": 6,
        "index": 5,
        "bit": 7,
        "expected_old": -3,
        "expected_new": 125,
    },
    {
        "name": "conv_w31_b7",
        "tensor": 6,
        "index": 31,
        "bit": 7,
        "expected_old": -20,
        "expected_new": 108,
    },
    {
        "name": "fc_w45024_b7",
        "tensor": 4,
        "index": 45024,
        "bit": 7,
        "expected_old": 48,
        "expected_new": -80,
    },
]

CMD_INFERENCE  = 0xA5
CMD_F1_FLIP    = 0xB1
CMD_F1_RESTORE = 0xB2
CMD_F1_STATUS  = 0xB3

RESP_INFERENCE  = 0x5A
RESP_F1_FLIP    = 0xC1
RESP_F1_RESTORE = 0xC2
RESP_F1_STATUS  = 0xC3

ERROR_PARAM  = 0xED
ERROR_RX     = 0xEE
ERROR_INVOKE = 0xEF

CHUNK_SIZE = 32
CHUNK_DELAY_S = 0.002
MAX_RETRIES = 3


def read_exact(ser, n, timeout_s=12.0):
    buf = bytearray()
    t0 = time.time()

    while len(buf) < n:
        waiting = ser.in_waiting

        if waiting:
            buf.extend(
                ser.read(
                    min(waiting, n - len(buf))
                )
            )
            t0 = time.time()

        elif time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"Timeout {len(buf)}/{n}"
            )

        else:
            time.sleep(0.001)

    return bytes(buf)


def check_header(h, expected):
    if h == ERROR_PARAM:
        raise RuntimeError("Nano: ERROR_PARAM 0xED")
    if h == ERROR_RX:
        raise RuntimeError("Nano: ERROR_RX 0xEE")
    if h == ERROR_INVOKE:
        raise RuntimeError("Nano: ERROR_INVOKE 0xEF")
    if h != expected:
        raise RuntimeError(
            f"Header 0x{h:02X}, attendu 0x{expected:02X}"
        )


def wait_ready(ser, timeout_s=5.0):
    print("Synchronisation Nano...")
    t0 = time.time()

    while time.time() - t0 < timeout_s:
        if ser.in_waiting:
            line = ser.readline().decode(
                errors="ignore"
            ).strip()

            if line:
                print("[Nano]", line)

            if "ERREUR FATALE" in line:
                raise RuntimeError(line)

            if line == "READY":
                print("READY recu.")
                return

        else:
            time.sleep(0.02)

    print("READY non recu ; test binaire direct.")


def send_chunks(ser, payload):
    for i in range(0, len(payload), CHUNK_SIZE):
        ser.write(
            payload[i:i + CHUNK_SIZE]
        )
        ser.flush()
        time.sleep(CHUNK_DELAY_S)


def send_status(ser):
    ser.write(bytes([CMD_F1_STATUS]))
    ser.flush()

    h = read_exact(ser, 1)[0]
    check_header(h, RESP_F1_STATUS)

    return read_exact(ser, 1)[0]


def send_flip(ser, tensor, index, bit):
    packet = bytearray()
    packet.append(CMD_F1_FLIP)
    packet.append(int(tensor))
    packet.extend(
        int(index).to_bytes(
            4,
            "little",
            signed=False
        )
    )
    packet.append(int(bit))

    ser.write(packet)
    ser.flush()

    h = read_exact(ser, 1)[0]
    check_header(h, RESP_F1_FLIP)

    tensor_r = read_exact(ser, 1)[0]
    index_r = int.from_bytes(
        read_exact(ser, 4),
        "little"
    )
    bit_r = read_exact(ser, 1)[0]
    offset_r = int.from_bytes(
        read_exact(ser, 4),
        "little"
    )

    old = int(
        np.frombuffer(
            read_exact(ser, 1),
            dtype=np.int8
        )[0]
    )

    new = int(
        np.frombuffer(
            read_exact(ser, 1),
            dtype=np.int8
        )[0]
    )

    return {
        "tensor": tensor_r,
        "index": index_r,
        "bit": bit_r,
        "offset": offset_r,
        "old": old,
        "new": new,
    }


def send_restore(ser):
    ser.write(bytes([CMD_F1_RESTORE]))
    ser.flush()

    h = read_exact(ser, 1)[0]
    check_header(h, RESP_F1_RESTORE)

    restored = int(
        np.frombuffer(
            read_exact(ser, 1),
            dtype=np.int8
        )[0]
    )

    return restored


def send_inference_once(ser, x):
    ser.write(bytes([CMD_INFERENCE]))
    ser.flush()

    payload = np.asarray(
        x,
        dtype=np.int8
    ).tobytes()

    send_chunks(
        ser,
        payload
    )

    h = read_exact(ser, 1)[0]
    check_header(h, RESP_INFERENCE)

    scores = np.frombuffer(
        read_exact(ser, 12),
        dtype=np.int8
    ).copy()

    pred = read_exact(ser, 1)[0]

    us = int.from_bytes(
        read_exact(ser, 4),
        "little"
    )

    return scores, pred, us


def with_retry(ser, fn, *args):
    last = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(ser, *args)

        except (TimeoutError, RuntimeError) as e:
            last = e
            msg = str(e)

            if (
                "ERROR_PARAM" in msg
                or "ERROR_INVOKE" in msg
            ):
                raise

            print(
                f"  [RETRY {attempt}/{MAX_RETRIES}] {msg}"
            )

            time.sleep(0.35)

            try:
                ser.reset_input_buffer()
                ser.reset_output_buffer()
            except Exception:
                pass

            time.sleep(0.15)

    raise RuntimeError(
        f"Echec apres {MAX_RETRIES} tentatives : {last}"
    )


def run_500(ser, X, y, mode_name, output_csv):
    rows = []
    preds = np.empty(len(X), dtype=np.int64)
    scores_all = np.empty((len(X), 12), dtype=np.int8)
    latencies = np.empty(len(X), dtype=np.int64)

    for i in range(len(X)):
        scores, pred, us = with_retry(
            ser,
            send_inference_once,
            X[i]
        )

        preds[i] = pred
        scores_all[i] = scores
        latencies[i] = us

        rows.append({
            "sample_order": i,
            "true_id": int(y[i]),
            "true_label": LABELS[int(y[i])],
            "prediction_id": int(pred),
            "prediction_label": LABELS[int(pred)],
            "correct": int(pred == int(y[i])),
            "latency_us": int(us),
            **{
                f"score_{j}": int(scores[j])
                for j in range(12)
            },
        })

        if (
            i == 0
            or (i + 1) % 50 == 0
            or i + 1 == len(X)
        ):
            print(
                f"[{i+1:3d}/{len(X)}] "
                f"{mode_name:16s} | "
                f"pred={LABELS[int(pred)]:12s} | "
                f"acc_cum={100*np.mean(preds[:i+1] == y[:i+1]):6.2f}%"
            )

    with Path(output_csv).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        w.writeheader()
        w.writerows(rows)

    acc = 100.0 * np.mean(preds == y)

    return {
        "preds": preds,
        "scores": scores_all,
        "latencies": latencies,
        "accuracy": acc,
        "mean_latency_ms": float(
            np.mean(latencies) / 1000.0
        ),
    }


parser = argparse.ArgumentParser(
    description=(
        "Campagne F1 runtime Nano : nominal + 3 fautes connues "
        "sur les 500 entrees."
    )
)

parser.add_argument(
    "--port",
    required=True
)
parser.add_argument(
    "--baseline",
    default="baseline_500.npz"
)
parser.add_argument(
    "--output-dir",
    default="resultats_runtime"
)

args = parser.parse_args()

base = np.load(args.baseline)

X = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)

if len(X) != 500:
    print(
        f"⚠ Attention : baseline contient {len(X)} entrees, pas 500."
    )

outdir = Path(args.output_dir)
outdir.mkdir(
    parents=True,
    exist_ok=True
)

summary_rows = []


with serial.Serial(
    args.port,
    115200,
    timeout=0.2
) as ser:

    time.sleep(1.5)
    wait_ready(ser)

    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    if send_status(ser) != 0:
        print("Faute active detectee au demarrage -> RESTORE")
        send_restore(ser)

    print()
    print("========================================")
    print(" F1 RUNTIME - NOMINAL 500")
    print("========================================")

    nominal = run_500(
        ser,
        X,
        y,
        "nominal",
        outdir / "f1_runtime_nominal_500.csv"
    )

    print()
    print(
        f"Nominal Nano runtime : "
        f"{nominal['accuracy']:.2f}% | "
        f"{nominal['mean_latency_ms']:.3f} ms"
    )

    summary_rows.append({
        "case": "nominal",
        "tensor": "",
        "index": "",
        "bit": "",
        "old": "",
        "new": "",
        "accuracy_percent": nominal["accuracy"],
        "delta_vs_runtime_nominal_points": 0.0,
        "prediction_changed_vs_runtime_nominal": 0,
        "scores_bit_exact_vs_runtime_nominal": 500,
        "mean_latency_ms": nominal["mean_latency_ms"],
    })

    for fault in FAULTS:
        print()
        print("========================================")
        print(
            f" F1 RUNTIME - {fault['name']}"
        )
        print("========================================")

        status = send_status(ser)

        if status != 0:
            raise RuntimeError(
                "Une faute est deja active avant FLIP."
            )

        info = send_flip(
            ser,
            fault["tensor"],
            fault["index"],
            fault["bit"]
        )

        print(
            f"FLIP OK | tensor={info['tensor']} "
            f"index={info['index']} bit={info['bit']} "
            f"offset={info['offset']} "
            f"{info['old']} -> {info['new']}"
        )

        if (
            info["old"] != fault["expected_old"]
            or info["new"] != fault["expected_new"]
        ):
            raise RuntimeError(
                f"Valeur FLIP inattendue pour {fault['name']}"
            )

        result = run_500(
            ser,
            X,
            y,
            fault["name"],
            outdir / f"f1_runtime_{fault['name']}_500.csv"
        )

        changed = int(
            np.sum(
                result["preds"]
                != nominal["preds"]
            )
        )

        exact_scores = int(
            np.sum(
                np.all(
                    result["scores"]
                    == nominal["scores"],
                    axis=1
                )
            )
        )

        print()
        print(
            f"{fault['name']} : "
            f"{result['accuracy']:.2f}% | "
            f"delta={result['accuracy']-nominal['accuracy']:+.2f} pts | "
            f"predΔ={changed}/500"
        )

        restored = send_restore(ser)

        if restored != fault["expected_old"]:
            raise RuntimeError(
                f"RESTORE inattendu : {restored}, "
                f"attendu {fault['expected_old']}"
            )

        if send_status(ser) != 0:
            raise RuntimeError(
                "Faute encore active apres RESTORE."
            )

        # Verification rapide apres RESTORE sur sample 0.
        restored_scores, restored_pred, _ = with_retry(
            ser,
            send_inference_once,
            X[0]
        )

        if not np.array_equal(
            restored_scores,
            nominal["scores"][0]
        ):
            raise RuntimeError(
                f"RESTORE non bit-exact apres {fault['name']}."
            )

        if restored_pred != nominal["preds"][0]:
            raise RuntimeError(
                f"RESTORE prediction differente apres {fault['name']}."
            )

        print(
            "RESTORE OK | sample 0 revenu exactement au nominal."
        )

        summary_rows.append({
            "case": fault["name"],
            "tensor": fault["tensor"],
            "index": fault["index"],
            "bit": fault["bit"],
            "old": info["old"],
            "new": info["new"],
            "accuracy_percent": result["accuracy"],
            "delta_vs_runtime_nominal_points": (
                result["accuracy"]
                - nominal["accuracy"]
            ),
            "prediction_changed_vs_runtime_nominal": changed,
            "scores_bit_exact_vs_runtime_nominal": exact_scores,
            "mean_latency_ms": result["mean_latency_ms"],
        })


summary_path = (
    outdir
    / "f1_runtime_resume.csv"
)

with summary_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )
    w.writeheader()
    w.writerows(summary_rows)


print()
print("========================================")
print(" F1 RUNTIME - RESUME FINAL")
print("========================================")

for r in summary_rows:
    print(
        f"{r['case']:18s} | "
        f"acc={float(r['accuracy_percent']):6.2f}% | "
        f"delta={float(r['delta_vs_runtime_nominal_points']):+6.2f} | "
        f"predΔ={int(r['prediction_changed_vs_runtime_nominal']):3d} | "
        f"lat={float(r['mean_latency_ms']):7.3f} ms"
    )

print()
print("Attendus historiques pre-fautes :")
print("  nominal          : 67.00 %")
print("  conv_w5_b7       : 55.80 %")
print("  conv_w31_b7      : 58.60 %")
print("  fc_w45024_b7     : 67.00 %")
print()
print("✅", summary_path)
