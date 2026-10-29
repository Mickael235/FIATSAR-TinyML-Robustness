# F1 runtime firmware

Generate `model_data.h` from `models/tiny_conv_int8.tflite` before compiling:

```bash
python ../../scripts/tools/10_tflite_to_model_data.py   --model ../../models/tiny_conv_int8.tflite   --output model_data.h
```

The included `f1_offsets.h` matches the baseline model SHA-256 documented in the project logs.
You can regenerate it with `scripts/faults/f1/f1_offsets.py`.
