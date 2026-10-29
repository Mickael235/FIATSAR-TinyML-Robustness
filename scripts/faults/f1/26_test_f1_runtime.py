import argparse
import time

import numpy as np
import serial


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


def read_exact(ser, n, timeout_s=10.0):
    buf = bytearray()
    t0 = time.time()

    while len(buf) < n:
        if ser.in_waiting:
            buf.extend(
                ser.read(
                    min(
                        ser.in_waiting,
                        n - len(buf)
                    )
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

    print(
        "READY non recu ; on poursuit avec STATUS."
    )


def check_header(h, expected):
    if h == ERROR_PARAM:
        raise RuntimeError(
            "Nano: ERROR_PARAM 0xED"
        )

    if h == ERROR_RX:
        raise RuntimeError(
            "Nano: ERROR_RX 0xEE"
        )

    if h == ERROR_INVOKE:
        raise RuntimeError(
            "Nano: ERROR_INVOKE 0xEF"
        )

    if h != expected:
        raise RuntimeError(
            f"Header 0x{h:02X}, attendu 0x{expected:02X}"
        )


def send_status(ser):
    ser.write(bytes([CMD_F1_STATUS]))
    ser.flush()

    h = read_exact(ser, 1)[0]
    check_header(
        h,
        RESP_F1_STATUS
    )

    return read_exact(
        ser,
        1
    )[0]


def send_flip(
    ser,
    tensor,
    index,
    bit
):
    packet = bytearray()

    packet.append(
        CMD_F1_FLIP
    )

    packet.append(
        tensor
    )

    packet.extend(
        int(index).to_bytes(
            4,
            "little",
            signed=False
        )
    )

    packet.append(
        bit
    )

    ser.write(
        packet
    )

    ser.flush()

    h = read_exact(
        ser,
        1
    )[0]

    check_header(
        h,
        RESP_F1_FLIP
    )

    tensor_r = read_exact(
        ser,
        1
    )[0]

    index_r = int.from_bytes(
        read_exact(
            ser,
            4
        ),
        "little"
    )

    bit_r = read_exact(
        ser,
        1
    )[0]

    offset_r = int.from_bytes(
        read_exact(
            ser,
            4
        ),
        "little"
    )

    old = int(
        np.frombuffer(
            read_exact(
                ser,
                1
            ),
            dtype=np.int8
        )[0]
    )

    new = int(
        np.frombuffer(
            read_exact(
                ser,
                1
            ),
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
    ser.write(
        bytes([
            CMD_F1_RESTORE
        ])
    )

    ser.flush()

    h = read_exact(
        ser,
        1
    )[0]

    check_header(
        h,
        RESP_F1_RESTORE
    )

    restored = int(
        np.frombuffer(
            read_exact(
                ser,
                1
            ),
            dtype=np.int8
        )[0]
    )

    return restored


def send_inference(
    ser,
    x
):
    ser.write(
        bytes([
            CMD_INFERENCE
        ])
    )

    ser.flush()

    payload = np.asarray(
        x,
        dtype=np.int8
    ).tobytes()

    for i in range(
        0,
        len(payload),
        32
    ):
        ser.write(
            payload[i:i+32]
        )
        ser.flush()
        time.sleep(0.002)

    h = read_exact(
        ser,
        1
    )[0]

    check_header(
        h,
        RESP_INFERENCE
    )

    scores = np.frombuffer(
        read_exact(
            ser,
            12
        ),
        dtype=np.int8
    ).copy()

    pred = read_exact(
        ser,
        1
    )[0]

    us = int.from_bytes(
        read_exact(
            ser,
            4
        ),
        "little"
    )

    return scores, pred, us


parser = argparse.ArgumentParser(
    description=(
        "Test unitaire F1 runtime : "
        "nominal -> FLIP -> inference -> RESTORE -> inference."
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
    "--sample",
    type=int,
    default=0
)

parser.add_argument(
    "--tensor",
    type=int,
    default=6
)

parser.add_argument(
    "--index",
    type=int,
    default=5
)

parser.add_argument(
    "--bit",
    type=int,
    default=7
)

args = parser.parse_args()

base = np.load(
    args.baseline
)

X = base[
    "X_int8"
].astype(np.int8)

if not (
    0 <= args.sample < len(X)
):
    raise ValueError(
        "sample hors plage"
    )

x = X[
    args.sample
]


with serial.Serial(
    args.port,
    115200,
    timeout=0.2
) as ser:

    time.sleep(1.5)

    wait_ready(
        ser
    )

    time.sleep(0.1)

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print()
    print(
        "===== STATUS INITIAL ====="
    )

    status0 = send_status(
        ser
    )

    print(
        "Faute active :",
        status0
    )

    print()
    print(
        "===== INFERENCE NOMINALE ====="
    )

    scores0, pred0, us0 = (
        send_inference(
            ser,
            x
        )
    )

    print(
        "Prediction :",
        pred0
    )

    print(
        "Scores     :",
        scores0.tolist()
    )

    print(
        "Latence    :",
        f"{us0/1000:.3f} ms"
    )

    print()
    print(
        "===== FLIP ====="
    )

    info = send_flip(
        ser,
        args.tensor,
        args.index,
        args.bit
    )

    print(
        f"tensor={info['tensor']} "
        f"index={info['index']} "
        f"bit={info['bit']} "
        f"offset={info['offset']} "
        f"{info['old']} -> {info['new']}"
    )

    print()
    print(
        "===== INFERENCE FAUTEE ====="
    )

    scores1, pred1, us1 = (
        send_inference(
            ser,
            x
        )
    )

    print(
        "Prediction :",
        pred1
    )

    print(
        "Scores     :",
        scores1.tolist()
    )

    print(
        "Scores modifies :",
        int(
            np.sum(
                scores0 != scores1
            )
        ),
        "/12"
    )

    print()
    print(
        "===== RESTORE ====="
    )

    restored = send_restore(
        ser
    )

    print(
        "Valeur restauree :",
        restored
    )

    print()
    print(
        "===== INFERENCE APRES RESTORE ====="
    )

    scores2, pred2, us2 = (
        send_inference(
            ser,
            x
        )
    )

    print(
        "Prediction :",
        pred2
    )

    print(
        "Scores     :",
        scores2.tolist()
    )

    print()
    print(
        "===== VERIFICATION ====="
    )

    print(
        "Nominal == Restore scores :",
        bool(
            np.array_equal(
                scores0,
                scores2
            )
        )
    )

    print(
        "Nominal == Restore pred   :",
        pred0 == pred2
    )

    print(
        "Faute change scores       :",
        not np.array_equal(
            scores0,
            scores1
        )
    )

    print(
        "Faute change prediction   :",
        pred0 != pred1
    )

    status_end = send_status(
        ser
    )

    print(
        "Faute active a la fin     :",
        status_end
    )

    if not np.array_equal(
        scores0,
        scores2
    ):
        raise RuntimeError(
            "RESTORE ne reproduit pas exactement le nominal."
        )

    if status_end != 0:
        raise RuntimeError(
            "Une faute reste active apres RESTORE."
        )

    print()
    print(
        "✅ TEST UNITAIRE F1 RUNTIME OK"
    )
