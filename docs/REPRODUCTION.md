# Reproduction notes

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The repository does not redistribute the complete Speech Commands dataset.

The fault scripts expect a prepared fixed test set / baseline NPZ generated with:

```bash
python scripts/data/01_prepare_testset_500.py
python scripts/data/02_baseline_500.py
```

Adapt input paths to your local dataset/model location.

## F1

```bash
python scripts/faults/f1/f1_offsets.py \
  --model models/tiny_conv_int8.tflite \
  --output firmware/f1_runtime/f1_offsets.h
```

Generate the Arduino header:

```bash
python scripts/tools/10_tflite_to_model_data.py \
  --model models/tiny_conv_int8.tflite \
  --output firmware/f1_runtime/model_data.h
```

Compile the F1 firmware and run:

```bash
python scripts/faults/f1/26_test_f1_runtime.py --port COM3 \
  --baseline baseline_500.npz \
  --sample 0 --tensor 6 --index 5 --bit 7
```

## F2

```bash
python scripts/faults/f2/14_f2_campaign.py \
  --mode bitflip --trials-per-layer 1000 --seed 42
```

The Arduino F2 path uses an instrumented model exposing FC logits plus a second Softmax model.

## F3

```bash
python scripts/faults/f3/18_f3_sweep.py --help
python scripts/faults/f3/19_f3_select_candidates.py --help
```

## F4

```bash
python scripts/faults/f4/24_f4_sweep.py --baseline baseline_500.npz
```

After uploading the F4 firmware:

```bash
python scripts/faults/f4/25_uart_f4_500.py \
  --port COM3 --baseline baseline_500.npz \
  --mode force --class-id 0 \
  --output f4_force_silence_nano.csv
```
