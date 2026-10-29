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
CMD_F3_FC = 0xA8

RESP_NOMINAL = 0x5A
RESP_F3 = 0x5C

ERROR_PARAM = 0xED
ERROR_RX = 0xEE
ERROR_INVOKE = 0xEF

CHUNK_SIZE = 32
CHUNK_DELAY_S = 0.002
MAX_RETRIES = 3


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
            buf.extend(ser.read(min(waiting, n - len(buf))))
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
        ser.write(payload[i:i + CHUNK_SIZE])
        ser.flush()
        time.sleep(CHUNK_DELAY_S)


def parse_header(ser, expected):
    h = read_exact(ser, 1)[0]

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


def send_nominal_once(ser, x):
    ser.write(bytes([CMD_NOMINAL]))
    ser.flush()
    send_chunks(ser, np.asarray(x, dtype=np.int8).tobytes())

    parse_header(ser, RESP_NOMINAL)

    scores = np.frombuffer(read_exact(ser, 12), dtype=np.int8).copy()
    pred = read_exact(ser, 1)[0]
    us = int.from_bytes(read_exact(ser, 4), "little")

    return scores, pred, us


def send_f3_once(ser, x, k):
    ser.write(bytes([
        CMD_F3_FC,
        k & 0xFF,
        (k >> 8) & 0xFF,
    ]))
    ser.flush()

    send_chunks(ser, np.asarray(x, dtype=np.int8).tobytes())

    parse_header(ser, RESP_F3)

    scores = np.frombuffer(read_exact(ser, 12), dtype=np.int8).copy()
    pred = read_exact(ser, 1)[0]
    us = int.from_bytes(read_exact(ser, 4), "little")
    returned_k = int.from_bytes(read_exact(ser, 2), "little")

    if returned_k != k:
        raise RuntimeError(f"k retourne={returned_k}, attendu={k}")

    return scores, pred, us


def with_retry(ser, fn, *args):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(ser, *args)
        except (TimeoutError, RuntimeError) as e:
            last_error = e
            msg = str(e)

            # Paramètre invalide ou Invoke : ce n'est pas un problème de transport.
            if "ERROR_PARAM" in msg or "ERROR_INVOKE" in msg:
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
        f"Echec apres {MAX_RETRIES} tentatives : {last_error}"
    )


def load_done(path):
    done = set()

    if not path.exists():
        return done

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            done.add(int(row["sample_order"]))

    return done


def append_row(path, fieldnames, row):
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            w.writeheader()

        w.writerow(row)


parser = argparse.ArgumentParser(
    description="Validation robuste F3 FullyConnected PC vs Nano sur 500 entrees."
)
parser.add_argument("--port", required=True)
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument("--model", default="model_original.tflite")
parser.add_argument("--baseline", default="baseline_500.npz")
parser.add_argument("--fc-tail", default="f2_tail_fc_softmax.tflite")
parser.add_argument("--k", type=int, default=3879)
parser.add_argument("--output", default="f3_fc_k3879_nano.csv")
args = parser.parse_args()

if not (0 <= args.k <= 4000):
    raise ValueError("k doit etre dans [0,4000].")

base = np.load(args.baseline)
Xq = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)
indices = base["indices"].astype(np.int64)
pc_nominal_scores = base["scores_int8"].astype(np.int8)
pc_nominal_preds = base["preds"].astype(np.int64)

n = len(Xq)


# ============================================================
# REFERENCE PC F3
# ============================================================

print("===== REFERENCE PC F3 =====")
print("k =", args.k)

full = tf.lite.Interpreter(
    model_path=args.model,
    experimental_preserve_all_tensors=True,
    num_threads=1,
)
full.allocate_tensors()

tail = tf.lite.Interpreter(
    model_path=args.fc_tail,
    num_threads=1,
)
tail.allocate_tensors()

full_in = full.get_input_details()[0]
tail_in = tail.get_input_details()[0]
tail_out = tail.get_output_details()[0]
zero_point = int(tail_in["quantization"][1])

pc_fault_scores = np.empty((n, 12), dtype=np.int8)
pc_fault_preds = np.empty(n, dtype=np.int64)

for i in range(n):
    full.set_tensor(full_in["index"], Xq[i:i+1])
    full.invoke()

    a = full.get_tensor(9)[0].copy()

    if args.k < 4000:
        a[args.k:] = np.int8(zero_point)

    tail.set_tensor(tail_in["index"], a.reshape(tail_in["shape"]))
    tail.invoke()

    s = tail.get_tensor(tail_out["index"])[0]
    pc_fault_scores[i] = s
    pc_fault_preds[i] = int(np.argmax(s))

pc_nom_acc = 100.0 * np.mean(pc_nominal_preds == y)
pc_fault_acc = 100.0 * np.mean(pc_fault_preds == y)

print(f"PC nominal accuracy : {pc_nom_acc:.2f}%")
print(f"PC F3 accuracy      : {pc_fault_acc:.2f}%")
print(f"PC delta            : {pc_fault_acc - pc_nom_acc:+.2f} pts")


# ============================================================
# CSV RESUMABLE
# ============================================================

out_path = Path(args.output)
done = load_done(out_path)

fieldnames = [
    "sample_order", "dataset_index", "true_label", "k",
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


# ============================================================
# NANO
# ============================================================

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

        _, test_pred, test_us = with_retry(
            ser,
            send_nominal_once,
            Xq[first_pending]
        )

        print(
            "✅ Communication OK | "
            f"pred={LABELS[test_pred]} | "
            f"{test_us/1000:.3f} ms"
        )
        print()

    processed = len(done)

    for i in range(n):
        if i in done:
            continue

        nano_nom_scores, nano_nom_pred, nano_nom_us = with_retry(
            ser,
            send_nominal_once,
            Xq[i]
        )

        # Petite marge entre les deux transactions.
        time.sleep(0.010)

        nano_fault_scores, nano_fault_pred, nano_fault_us = with_retry(
            ser,
            send_f3_once,
            Xq[i],
            args.k
        )

        pc_nom_pred = int(pc_nominal_preds[i])
        pc_f_pred = int(pc_fault_preds[i])

        base_correct_pc = int(pc_nom_pred == int(y[i]))
        fault_correct_pc = int(pc_f_pred == int(y[i]))
        base_correct_nano = int(nano_nom_pred == int(y[i]))
        fault_correct_nano = int(nano_fault_pred == int(y[i]))

        row = {
            "sample_order": i,
            "dataset_index": int(indices[i]),
            "true_label": LABELS[int(y[i])],
            "k": args.k,
            "pc_nominal_pred": LABELS[pc_nom_pred],
            "pc_fault_pred": LABELS[pc_f_pred],
            "nano_nominal_pred": LABELS[nano_nom_pred],
            "nano_fault_pred": LABELS[nano_fault_pred],
            "pc_fault_changed": int(pc_nom_pred != pc_f_pred),
            "nano_fault_changed": int(nano_nom_pred != nano_fault_pred),
            "pc_new_error": int(
                base_correct_pc == 1 and fault_correct_pc == 0
            ),
            "nano_new_error": int(
                base_correct_nano == 1 and fault_correct_nano == 0
            ),
            "pc_new_critical": int(
                critical(y[i], pc_nom_pred) == 0
                and critical(y[i], pc_f_pred) == 1
            ),
            "nano_new_critical": int(
                critical(y[i], nano_nom_pred) == 0
                and critical(y[i], nano_fault_pred) == 1
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


# ============================================================
# RESUME FINAL
# ============================================================

with out_path.open("r", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

rows = sorted(rows, key=lambda r: int(r["sample_order"]))

if len(rows) != n:
    print()
    print(
        f"⚠ Campagne partielle sauvegardee : {len(rows)}/{n}. "
        "Relance la meme commande pour reprendre."
    )
    raise SystemExit(0)

nano_nom_correct = [
    LABELS.index(r["nano_nominal_pred"])
    == LABELS.index(r["true_label"])
    for r in rows
]
nano_fault_correct = [
    LABELS.index(r["nano_fault_pred"])
    == LABELS.index(r["true_label"])
    for r in rows
]

nano_nom_acc = 100.0 * np.mean(nano_nom_correct)
nano_fault_acc = 100.0 * np.mean(nano_fault_correct)

print()
print("===== F3 FC PC ↔ NANO =====")
print(f"k                         : {args.k}/4000")
print(f"PC nominal accuracy       : {pc_nom_acc:.2f}%")
print(f"PC F3 accuracy            : {pc_fault_acc:.2f}%")
print(f"PC delta                  : {pc_fault_acc - pc_nom_acc:+.2f} pts")
print()
print(f"Nano nominal accuracy     : {nano_nom_acc:.2f}%")
print(f"Nano F3 accuracy          : {nano_fault_acc:.2f}%")
print(f"Nano delta                : {nano_fault_acc - nano_nom_acc:+.2f} pts")
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
