# Original submission

Preserved exactly as submitted, for comparison. **None of it runs** — see
[`../docs/model-changes.md`](../docs/model-changes.md) for the ten defects.

| File | What it is |
|---|---|
| `capstone1_colab.py` | The Colab export. Starts with `!pip install`, so it is not valid Python. |
| `output_prices1.csv` | Model 1 output. Note `price` = 327.76 against a base of 10. |
| `output_prices2.csv`, `output_prices3.csv` | Model 2 and 3 output — one row per day, all 14 lots combined. |
| `parking_stream.csv` | Model 1 input. `competitor_counts` = 1312 on every row. |
| `parking_stream_model2.csv` | Model 2 input; the source of `../data/parking_dataset.csv`. |
| `*.png` | Figures from the original run. |

These files are the evidence behind the defect report — they are what the
original code actually produced, not a reconstruction.
