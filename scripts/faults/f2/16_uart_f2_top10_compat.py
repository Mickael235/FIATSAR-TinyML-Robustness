import argparse
import csv
import time

import numpy as np
import serial

LABELS = [
    "_silence_", "_unknown_", "yes", "no", "up", "down",
    "left", "right", "on", "off", "stop", "go"
]

CMD_NOMINAL = 0xA5
CMD_F2_BITFLIP = 0xA6
CMD_F2_ZERO = 0xA7

RESP_NOMINAL = 0x5A
RESP_F2 = 0x5B

ERROR_PARAM = 0xED
ERROR_RX = 0xEE
ERROR_INVOKE = 0xEF


def read_exact(ser, n, timeout_s=5.0):
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
                f"Timeout reception Nano : {len(buf)}/{n} octets"
            )
        else:
            time.sleep(0.001)

    return bytes(buf)


def startup_sync(ser, timeout_s=4.0):
    """
    Sur Nano 33 BLE / USB CDC, l'ouverture du port ne redemarre pas
    forcement la carte. READY peut donc avoir ete emis AVANT que
    Python ouvre COM3.

    On lit les logs s'ils sont presents, mais READY n'est plus
    obligatoire. La vraie validation sera la premiere reponse binaire
    0x5A a une inference nominale.
    """
    print("Synchronisation avec la Nano...")

    t0 = time.time()
    saw_ready = False
    text_lines = []

    while time.time() - t0 < timeout_s:
        if ser.in_waiting:
            raw = ser.readline()

            try:
                line = raw.decode(
                    errors="ignore"
                ).strip()
            except Exception:
                line = ""

            if line:
                text_lines.append(line)
                print("[Nano]", line)

                if "ERREUR FATALE" in line:
                    raise RuntimeError(
                        "Le firmware Arduino signale une erreur fatale : "
                        + line
                    )

                if line == "READY":
                    saw_ready = True
                    break
        else:
            time.sleep(0.02)

    if saw_ready:
        print("READY recu.")
    else:
        print(
            "READY non recu, mais ce n'est pas bloquant sur Nano 33 BLE : "
            "il a probablement ete emis avant l'ouverture de COM3."
        )

    # Nettoyage de ce qui resterait des logs texte de demarrage.
    time.sleep(0.15)
    ser.reset_input_buffer()
    ser.reset_output_buffer()


def send_chunks(ser, payload):
    for i in range(0, len(payload), 64):
        ser.write(payload[i:i+64])
        ser.flush()
        time.sleep(0.001)


def parse_header(ser, expected):
    h = read_exact(ser, 1, timeout_s=8.0)[0]

    if h == ERROR_PARAM:
        raise RuntimeError(
            "Nano: parametre F2 invalide (0xED)"
        )
    if h == ERROR_RX:
        raise RuntimeError(
            "Nano: erreur reception (0xEE)"
        )
    if h == ERROR_INVOKE:
        raise RuntimeError(
            "Nano: erreur Invoke (0xEF)"
        )
    if h != expected:
        raise RuntimeError(
            f"Header inattendu 0x{h:02X}, "
            f"attendu 0x{expected:02X}"
        )


def send_nominal(ser, x):
    ser.write(bytes([CMD_NOMINAL]))
    ser.flush()

    send_chunks(
        ser,
        np.asarray(x, dtype=np.int8).tobytes()
    )

    parse_header(ser, RESP_NOMINAL)

    scores = np.frombuffer(
        read_exact(ser, 12),
        dtype=np.int8
    ).copy()

    pred = read_exact(ser, 1)[0]

    us = int.from_bytes(
        read_exact(ser, 4),
        "little",
        signed=False
    )

    return scores, pred, us


def send_f2(ser, x, mode, index, bit):
    if mode == "bitflip":
        cmd = CMD_F2_BITFLIP
        bit_byte = bit

    elif mode == "zero":
        cmd = CMD_F2_ZERO
        bit_byte = 0

    else:
        raise ValueError(mode)

    ser.write(
        bytes([
            cmd,
            int(index),
            int(bit_byte),
        ])
    )
    ser.flush()

    send_chunks(
        ser,
        np.asarray(x, dtype=np.int8).tobytes()
    )

    parse_header(ser, RESP_F2)

    scores = np.frombuffer(
        read_exact(ser, 12),
        dtype=np.int8
    ).copy()

    pred = read_exact(ser, 1)[0]

    us = int.from_bytes(
        read_exact(ser, 4),
        "little",
        signed=False
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

    return scores, pred, us, old, new


parser = argparse.ArgumentParser(
    description="Rejoue sur Nano les TOP fautes F2 FullyConnected_logits."
)
parser.add_argument("--port", required=True)
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument(
    "--baseline",
    default="baseline_500.npz"
)
parser.add_argument(
    "--top10",
    default="f2_campaign_top10.csv"
)
parser.add_argument(
    "--output",
    default="f2_top10_nano.csv"
)
args = parser.parse_args()

base = np.load(args.baseline)
Xq = base["X_int8"].astype(np.int8)
y = base["y"].astype(np.int64)
indices = base["indices"].astype(np.int64)

with open(
    args.top10,
    "r",
    newline="",
    encoding="utf-8-sig"
) as f:
    rows = list(csv.DictReader(f))

jobs = [
    r for r in rows
    if r["layer"] == "FullyConnected_logits"
]

if not jobs:
    raise RuntimeError(
        "Aucune faute FullyConnected_logits dans le top."
    )

print(
    f"{len(jobs)} faute(s) FC a rejouer."
)

results = []

with serial.Serial(
    args.port,
    args.baud,
    timeout=0.2
) as ser:

    # Laisse le temps a l'USB CDC de se stabiliser.
    time.sleep(1.5)

    startup_sync(
        ser,
        timeout_s=4.0
    )

    # --------------------------------------------------------
    # PROBE NOMINALE
    # --------------------------------------------------------
    # C'est elle qui prouve reellement que le firmware F2 est
    # vivant, meme si READY avait ete emis avant l'ouverture.
    probe_sample = int(jobs[0]["sample_order"])

    print()
    print(
        f"Test de communication nominale sur sample={probe_sample}..."
    )

    try:
        probe_scores, probe_pred, probe_us = send_nominal(
            ser,
            Xq[probe_sample]
        )
    except Exception as e:
        raise RuntimeError(
            "La Nano ne repond pas au protocole binaire F2. "
            "Verifie que le firmware tiny_conv_uart_f2_fc est bien "
            "televerse et que le Moniteur serie Arduino est ferme. "
            f"Erreur originale : {e}"
        ) from e

    print(
        "✅ Communication F2 OK | "
        f"prediction={LABELS[probe_pred]} | "
        f"latence={probe_us/1000:.3f} ms"
    )
    print()

    # --------------------------------------------------------
    # TOP 10
    # --------------------------------------------------------
    for r in jobs:
        rank = int(r["rank"])
        mode = r["mode"]
        sample = int(r["sample_order"])
        idx = int(r["activation_index"])
        bit = int(r["bit"])

        nominal_scores, nominal_pred, nominal_us = (
            send_nominal(
                ser,
                Xq[sample]
            )
        )

        (
            fault_scores,
            fault_pred,
            fault_us,
            nano_old,
            nano_new,
        ) = send_f2(
            ser,
            Xq[sample],
            mode,
            idx,
            bit,
        )

        pc_base = r["base_pred"]
        pc_fault = r["fault_pred"]

        nano_base_name = LABELS[nominal_pred]
        nano_fault_name = LABELS[fault_pred]

        fault_match = int(
            nano_fault_name == pc_fault
        )

        transition_match = int(
            nano_base_name == pc_base
            and nano_fault_name == pc_fault
        )

        nano_changed = int(
            nominal_pred != fault_pred
        )

        true_id = int(y[sample])

        nano_new_error = int(
            nominal_pred == true_id
            and fault_pred != true_id
        )

        result = {
            "rank": rank,
            "mode": mode,
            "sample_order": sample,
            "dataset_index": int(indices[sample]),
            "true_label": LABELS[true_id],
            "activation_index": idx,
            "bit": bit,
            "pc_old_int8": int(r["old_int8"]),
            "pc_new_int8": int(r["new_int8"]),
            "nano_old_int8": nano_old,
            "nano_new_int8": nano_new,
            "pc_base_pred": pc_base,
            "pc_fault_pred": pc_fault,
            "nano_base_pred": nano_base_name,
            "nano_fault_pred": nano_fault_name,
            "nano_prediction_changed": nano_changed,
            "nano_new_error": nano_new_error,
            "fault_prediction_match_pc": fault_match,
            "full_transition_match_pc": transition_match,
            "nano_nominal_us": nominal_us,
            "nano_fault_us": fault_us,
        }

        results.append(result)

        print(
            f"#{rank:2d} {mode:7s} "
            f"sample={sample:3d} idx={idx:2d} "
            f"bit={bit:2d} | "
            f"PC {pc_base}->{pc_fault} | "
            f"Nano {nano_base_name}->{nano_fault_name} | "
            f"logit {nano_old}->{nano_new} | "
            f"fault_match={fault_match}"
        )

with open(
    args.output,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:
    w = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )
    w.writeheader()
    w.writerows(results)

n = len(results)

print()
print("===== F2 TOP N - PC ↔ NANO =====")
print(
    "Predictions fautees identiques : "
    f"{sum(r['fault_prediction_match_pc'] for r in results)}/{n}"
)
print(
    "Transitions completes identiques : "
    f"{sum(r['full_transition_match_pc'] for r in results)}/{n}"
)
print(
    "Fautes changeant la prediction Nano : "
    f"{sum(r['nano_prediction_changed'] for r in results)}/{n}"
)
print(
    "Nouvelles erreurs Nano : "
    f"{sum(r['nano_new_error'] for r in results)}/{n}"
)
print(
    "Latence nominale moyenne : "
    f"{np.mean([r['nano_nominal_us'] for r in results])/1000:.3f} ms"
)
print(
    "Latence F2 moyenne : "
    f"{np.mean([r['nano_fault_us'] for r in results])/1000:.3f} ms"
)
print("✅", args.output)
