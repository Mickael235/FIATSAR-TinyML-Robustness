import argparse
import csv
import time
from pathlib import Path

import numpy as np
import serial
import tensorflow as tf

LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

CRITICAL_PAIRS = {
    ("stop", "go"), ("go", "stop"),
    ("left", "right"), ("right", "left"),
    ("on", "off"), ("off", "on"),
}

CMD_NOMINAL = 0xA5
CMD_F3_CONV = 0xA9

RESP_NOMINAL = 0x5A
RESP_F3_CONV = 0x5D

ERROR_RX = 0xEE
ERROR_INVOKE = 0xEF

CHUNK_SIZE = 32
CHUNK_DELAY_S = 0.002
MAX_RETRIES = 3
F3_K = 79


def critical(true_id, pred_id):
    t = LABELS[int(true_id)]
    p = LABELS[int(pred_id)]

    return int(
        (t, p) in CRITICAL_PAIRS
        or (
            t not in ("_silence_", "_unknown_")
            and p == "_silence_"
        )
    )


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
            raise TimeoutError(f"Timeout {len(buf)}/{n}")
        else:
            time.sleep(0.001)

    return bytes(buf)


def startup_sync(ser, timeout_s=5.0):
    print("Synchronisation Nano...")
    t0 = time.time()
    saw_ready = False

    while time.time() - t0 < timeout_s:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()

            if line:
                print("[Nano]", line)

            if "ERREUR FATALE" in line:
                raise RuntimeError(line)

            if line == "READY":
                saw_ready = True
                break
        else:
            time.sleep(0.02)

    if saw_ready:
        print("READY recu.")
    else:
        print("READY non recu ; test binaire direct.")

    time.sleep(0.10)
    ser.reset_input_buffer()
    ser.reset_output_buffer()


def send_chunks(ser, payload):
    for i in range(0, len(payload), CHUNK_SIZE):
        ser.write(payload[i:i+CHUNK_SIZE])
        ser.flush()
        time.sleep(CHUNK_DELAY_S)


def parse_header(ser, expected):
    h = read_exact(ser, 1)[0]

    if h == ERROR_RX:
        raise RuntimeError("Nano: ERROR_RX 0xEE")
    if h == ERROR_INVOKE:
        raise RuntimeError("Nano: ERROR_INVOKE 0xEF")
    if h != expected:
        raise RuntimeError(
            f"Header 0x{h:02X}, attendu 0x{expected:02X}"
        )


def send_once(ser, x, fault=False):
    command = CMD_F3_CONV if fault else CMD_NOMINAL
    expected = RESP_F3_CONV if fault else RESP_NOMINAL

    ser.write(bytes([command]))
    ser.flush()

    send_chunks(
        ser,
        np.asarray(x, dtype=np.int8).tobytes()
    )

    parse_header(ser, expected)

    scores = np.frombuffer(
        read_exact(ser, 12),
        dtype=np.int8
    ).copy()

    pred = read_exact(ser, 1)[0]
    us = int.from_bytes(read_exact(ser, 4), "little")

    if fault:
        returned_k = read_exact(ser, 1)[0]
        if returned_k != F3_K:
            raise RuntimeError(
                f"k retourne={returned_k}, attendu={F3_K}"
            )

    return scores, pred, us


def with_retry(ser, fn, *args, **kwargs):
    last = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(ser, *args, **kwargs)
        except (TimeoutError, RuntimeError) as e:
            last = e
            msg = str(e)

            if "ERROR_INVOKE" in msg:
                raise

            print(f"  [RETRY {attempt}/{MAX_RETRIES}] {msg}")
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


def load_done(path):
    done = set()

    if not path.exists():
        return done

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            done.add(int(r["sample_order"]))

    return done


def append_row(path, fieldnames, row):
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            w.writeheader()

        w.writerow(row)


parser = argparse.ArgumentParser(
    description="Validation F3 Conv2D k=79/80 PC vs Nano."
)
parser.add_argument("--port", required=True)
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--fault-model", default="f3_conv_k79.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--output", default="f3_conv_k79_nano.csv")
args = parser.parse_args()

base = np.load(args.baseline)

Xq = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)
indices = base["indices"].astype(np.int64)
pc_nom_scores = base["scores_int8"].astype(np.int8)
pc_nom_preds = base["preds"].astype(np.int64)

n = len(Xq)


# ============================================================
# REFERENCE PC FAUTEE
# ============================================================

print("===== REFERENCE PC F3 CONV =====")
print("k = 79/80")

fault_interp = tf.lite.Interpreter(
    model_path=args.fault_model,
    num_threads=1
)
fault_interp.allocate_tensors()

fi = fault_interp.get_input_details()[0]
fo = fault_interp.get_output_details()[0]

pc_fault_scores = np.empty((n, 12), dtype=np.int8)
pc_fault_preds = np.empty(n, dtype=np.int64)

for i in range(n):
    fault_interp.set_tensor(fi["index"], Xq[i:i+1])
    fault_interp.invoke()
    s = fault_interp.get_tensor(fo["index"])[0]
    pc_fault_scores[i] = s
    pc_fault_preds[i] = int(np.argmax(s))

pc_nom_acc = 100.0 * np.mean(pc_nom_preds == y)
pc_fault_acc = 100.0 * np.mean(pc_fault_preds == y)

pc_base_correct = pc_nom_preds == y
pc_fault_correct = pc_fault_preds == y

pc_new_error = pc_base_correct & (~pc_fault_correct)

pc_base_crit = np.array(
    [critical(t, p) for t, p in zip(y, pc_nom_preds)],
    dtype=bool
)
pc_fault_crit = np.array(
    [critical(t, p) for t, p in zip(y, pc_fault_preds)],
    dtype=bool
)
pc_new_crit = (~pc_base_crit) & pc_fault_crit

print(f"PC nominal accuracy   : {pc_nom_acc:.2f}%")
print(f"PC F3 Conv accuracy   : {pc_fault_acc:.2f}%")
print(f"PC delta              : {pc_fault_acc-pc_nom_acc:+.2f} pts")
print(f"PC predictions change : {int(np.sum(pc_nom_preds != pc_fault_preds))}")
print(f"PC nouvelles erreurs  : {int(np.sum(pc_new_error))}")
print(f"PC nouvelles critiques: {int(np.sum(pc_new_crit))}")


out_path = Path(args.output)
done = load_done(out_path)

fieldnames = [
    "sample_order", "dataset_index", "true_label",
    "pc_nominal_pred", "pc_fault_pred",
    "nano_nominal_pred", "nano_fault_pred",
    "pc_fault_changed", "nano_fault_changed",
    "pc_new_error", "nano_new_error",
    "pc_new_critical", "nano_new_critical",
    "fault_prediction_match_pc",
    "full_transition_match_pc",
    "fault_scores_bit_exact_pc",
    "nano_nominal_us", "nano_fault_us",
]

print()
print(f"Deja sauvegardes : {len(done)}/{n}")


with serial.Serial(
    args.port,
    args.baud,
    timeout=0.2
) as ser:

    time.sleep(1.5)
    startup_sync(ser)

    first_pending = next(
        (i for i in range(n) if i not in done),
        None
    )

    if first_pending is not None:
        print()
        print("Test nominal...")

        _, pred, us = with_retry(
            ser,
            send_once,
            Xq[first_pending],
            False
        )

        print(
            "✅ Communication OK | "
            f"pred={LABELS[pred]} | {us/1000:.3f} ms"
        )
        print()

    processed = len(done)

    for i in range(n):
        if i in done:
            continue

        nano_nom_scores, nano_nom_pred, nano_nom_us = with_retry(
            ser,
            send_once,
            Xq[i],
            False
        )

        time.sleep(0.010)

        nano_fault_scores, nano_fault_pred, nano_fault_us = with_retry(
            ser,
            send_once,
            Xq[i],
            True
        )

        true_id = int(y[i])
        pc_nom_pred = int(pc_nom_preds[i])
        pc_f_pred = int(pc_fault_preds[i])

        base_correct_nano = int(nano_nom_pred == true_id)
        fault_correct_nano = int(nano_fault_pred == true_id)

        row = {
            "sample_order": i,
            "dataset_index": int(indices[i]),
            "true_label": LABELS[true_id],
            "pc_nominal_pred": LABELS[pc_nom_pred],
            "pc_fault_pred": LABELS[pc_f_pred],
            "nano_nominal_pred": LABELS[nano_nom_pred],
            "nano_fault_pred": LABELS[nano_fault_pred],
            "pc_fault_changed": int(pc_nom_pred != pc_f_pred),
            "nano_fault_changed": int(nano_nom_pred != nano_fault_pred),
            "pc_new_error": int(
                pc_nom_pred == true_id
                and pc_f_pred != true_id
            ),
            "nano_new_error": int(
                base_correct_nano == 1
                and fault_correct_nano == 0
            ),
            "pc_new_critical": int(
                critical(true_id, pc_nom_pred) == 0
                and critical(true_id, pc_f_pred) == 1
            ),
            "nano_new_critical": int(
                critical(true_id, nano_nom_pred) == 0
                and critical(true_id, nano_fault_pred) == 1
            ),
            "fault_prediction_match_pc": int(
                nano_fault_pred == pc_f_pred
            ),
            "full_transition_match_pc": int(
                nano_nom_pred == pc_nom_pred
                and nano_fault_pred == pc_f_pred
            ),
            "fault_scores_bit_exact_pc": int(
                np.array_equal(
                    nano_fault_scores,
                    pc_fault_scores[i]
                )
            ),
            "nano_nominal_us": nano_nom_us,
            "nano_fault_us": nano_fault_us,
        }

        append_row(out_path, fieldnames, row)
        done.add(i)
        processed += 1

        if processed == 1 or processed % 25 == 0 or processed == n:
            print(
                f"[{processed:3d}/{n}] "
                f"PC {LABELS[pc_nom_pred]}->{LABELS[pc_f_pred]} | "
                f"Nano {LABELS[nano_nom_pred]}->{LABELS[nano_fault_pred]} | "
                f"fault_match={row['fault_prediction_match_pc']}"
            )


with out_path.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

rows = sorted(rows, key=lambda r: int(r["sample_order"]))

if len(rows) != n:
    print()
    print(
        f"⚠ Campagne partielle : {len(rows)}/{n}. "
        "Relance la meme commande."
    )
    raise SystemExit(0)

nano_nom_acc = 100.0 * np.mean([
    LABELS.index(r["nano_nominal_pred"])
    == LABELS.index(r["true_label"])
    for r in rows
])

nano_fault_acc = 100.0 * np.mean([
    LABELS.index(r["nano_fault_pred"])
    == LABELS.index(r["true_label"])
    for r in rows
])

print()
print("===== F3 CONV PC ↔ NANO =====")
print("k                         : 79/80")
print(f"PC nominal accuracy       : {pc_nom_acc:.2f}%")
print(f"PC F3 accuracy            : {pc_fault_acc:.2f}%")
print(f"PC delta                  : {pc_fault_acc-pc_nom_acc:+.2f} pts")
print()
print(f"Nano nominal accuracy     : {nano_nom_acc:.2f}%")
print(f"Nano F3 accuracy          : {nano_fault_acc:.2f}%")
print(f"Nano delta                : {nano_fault_acc-nano_nom_acc:+.2f} pts")
print()
print(
    "Predictions fautees PC/Nano : "
    f"{sum(int(r['fault_prediction_match_pc']) for r in rows)}/{n}"
)
print(
    "Transitions completes       : "
    f"{sum(int(r['full_transition_match_pc']) for r in rows)}/{n}"
)
print(
    "Scores fautes bit-exacts    : "
    f"{sum(int(r['fault_scores_bit_exact_pc']) for r in rows)}/{n}"
)
print(
    "Nouvelles erreurs Nano      : "
    f"{sum(int(r['nano_new_error']) for r in rows)}"
)
print(
    "Nouvelles critiques Nano    : "
    f"{sum(int(r['nano_new_critical']) for r in rows)}"
)
print(
    "Latence nominale moyenne    : "
    f"{np.mean([int(r['nano_nominal_us']) for r in rows])/1000:.3f} ms"
)
print(
    "Latence F3 moyenne          : "
    f"{np.mean([int(r['nano_fault_us']) for r in rows])/1000:.3f} ms"
)
print("✅", out_path)
