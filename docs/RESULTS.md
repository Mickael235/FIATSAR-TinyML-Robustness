# Experimental results

## Baseline used by the fault campaigns

- Fixed test set: 500 quantized inputs.
- Desktop TFLite nominal accuracy: 73.4%.
- Nano nominal accuracy in the fault campaign: 67.0%.
- Typical Nano inference latency: about 85 ms.

## F1 — weight bit-flip

| Case | PC | Nano runtime | Nano Δ | Predictions changed |
|---|---:|---:|---:|---:|
| Nominal | 73.4% | 67.0% | 0.0 pt | 0 / 500 |
| Conv w5 bit7 | 63.2% | 55.8% | -11.2 pt | 156 / 500 |
| Conv w31 bit7 | 65.0% | 58.6% | -8.4 pt | 143 / 500 |
| FC w45024 bit7 | 72.8% | 67.0% | 0.0 pt | 0 / 500 |

The runtime implementation reproduces the historical pre-faulted Nano accuracies exactly.

## F2 — activation corruption

4,000 random fault trials were run:

- 1,000 Conv bit-flips;
- 1,000 FC-logit bit-flips;
- 1,000 Conv zeroing trials;
- 1,000 FC-logit zeroing trials.

Overall bit-flip campaign:

- Conv: prediction changed in ~0.9% of sampled trials;
- FC logits: prediction changed in ~16.2% of sampled trials.

Bit sensitivity:

- FC bit 7: 50.0% of sampled injections changed the prediction;
- Conv bit 7: ~3.64%.

Nano replay of selected critical cases:

- faulted prediction matches PC: 10/10;
- complete nominal→fault transition matches: 9/10;
- faults changing the Nano prediction: 10/10.

## F3 — truncated accumulation

Late candidates demonstrate that even near-complete accumulation can still alter decisions.

Examples:

- FC `k=3879/4000`: 96.975% of terms processed;
- FC `k=3995/4000`: only 5 terms skipped;
- Conv `k=79/80`: one contribution skipped.

The current implementation reproduces the mathematical effect on the Nano but is not yet a physical instruction skip.

## F4 — output corruption

| Nano scenario | Accuracy |
|---|---:|
| Nominal | 67.0% |
| Truncate `k=11/12` | 66.2% |
| Force `go` | 12.2% |
| Force `_silence_` | 8.4% |

For forced `_silence_`:

- 294 new Nano errors;
- 406 new critical confusions;
- faulted prediction agreement PC/Nano: 500/500.

## Adversarial validation subset

The public snapshot also includes a small 50-sample-per-condition UART validation subset for:

- clean;
- additive-noise conditions;
- FGSM ε=0.5;
- PGD ε=0.5;
- original and adversarially trained TinyConv.

The complete adversarial training workspace is not published yet.
