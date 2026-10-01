# What changed, and why

The original submission lives unmodified at [`legacy/capstone1_colab.py`](../legacy/capstone1_colab.py).
This file records every defect found in it and what replaced it, so the two
versions can be compared honestly.

Each claim below was verified against the committed artefacts of the original
run (`output_prices1.csv`, `parking_stream_model2.csv`), not inferred from
reading the code.

---

## 1. The script could not run

`capstone1.py` opened with `!pip install …`, which is notebook syntax, not
Python. The file raised `SyntaxError` on import. It also called
`from google.colab import files` and `files.upload()`, so it could only ever run
inside one Colab session, against a `dataset.csv` that was never committed.

**Now:** a plain package under `src/parking_pricing/`, a committed dataset at
`data/parking_dataset.csv`, and `python -m parking_pricing.cli` as the entry
point. Nothing is notebook-specific.

## 2. `datetime` was shadowed

```python
import datetime                  # line 17
from datetime import datetime    # line 22  <- rebinds the name
...
window=pw.temporal.tumbling(datetime.timedelta(days=1))   # line 141
```

After line 22, `datetime` is the *class*, which has no `timedelta` attribute, so
line 141 raises `AttributeError`. In Colab this survived only because cells were
executed out of order.

**Now:** no wildcard or shadowing imports.

## 3. Proximity features counted each lot as its own neighbour

`compute_proximity_features_fast` looped over all 18,368 **rows**. Each lot
appears once per timestamp, so a lot's own 1,312 repeated rows were treated as
nearby parking lots.

The committed output proves it: `competitor_counts` is **1312** for every row —
exactly the number of timesteps per lot — and `nearest_distances` is ~5 m, the
GPS jitter between a lot and itself. With 14 lots, no lot can have more than 13
competitors.

**Now:** `features.proximity_features` runs on one row per lot
(`data.lot_locations`), fills the distance-matrix diagonal with infinity, and
*raises* if handed a per-timestamp frame. Real values: 0–5 competitors per lot,
nearest distances from 1.3 m (genuinely co-located lots) to 1,560 km (the
`20.0°N, 78.0°E` outlier).

Covered by `test_lot_is_never_its_own_competitor`,
`test_proximity_rejects_per_timestamp_frame`, `test_competitor_counts_are_plausible`.

## 4. Model 1 priced an empty lot highest, without limit

```python
price = 10 + 5 * (cap / (occ_min + 0.1))
```

Price is *inversely* proportional to occupancy — the emptier the lot, the more
it charges — and the `+0.1` guard means a lot that ever empties approaches
`10 + 5·cap/0.1`. The committed `output_prices1.csv` shows **327.76** against a
base price of 10.

**Now:** `price = base · (1 + α · occupancy_rate)`, monotonically increasing,
and every model's output is clamped to `[0.5×, 2×]` base. Covered by
`test_model1_increases_with_occupancy` and `test_model1_stays_in_band`.

> **Note on levels.** Model 1 is an upward-only baseline (10 → 20 as a lot
> fills), while Models 2 and 3 are centred on the base price, so their means sit
> lower (≈8.5). They answer different questions and are not calibrated to one
> another; compare each model's *shape*, not its level.

## 5. There was no per-lot pricing

Every window used `instance=pw.this.day` — the calendar day alone. All 14 lots
were summed into one aggregate and emitted as a single price per day. A pricing
engine for 14 lots produced 73 numbers total, none of them attributable to a lot.

**Now:** one price per lot per timestamp (18,282 rows), with `lot_id` assigned
from coordinates in `data.assign_lot_ids`. Covered by
`test_pipeline_prices_one_row_per_lot_timestamp`.

## 6. Demand drivers were summed on incompatible scales

Model 2 added `occupancy_rate` (0–1), `QueueLength` (0–10), `TrafficLevel`
(0–2), `IsSpecialDay` (0–1) and `VehicleTypeWeight` (1.0–2.0) with hand-set
weights. Because the scales differ by an order of magnitude, the *units*
dominated the weighting: the 0.5 queue weight outweighed the 1.0 occupancy
weight by roughly 5×, the opposite of what the numbers suggest.

**Now:** `features.add_demand_features` maps every driver onto 0–1 first, so the
configured weights mean what they say. Normalisation bounds are derived from the
weights (`models.demand_bounds`), not from the observed data, so a row prices
identically whether it arrives alone or in a batch — necessary for a streaming
system and covered by `test_model2_is_deterministic_per_row`.

## 7. The schema contradicted the data

```python
class ParkingSchema(pw.Schema):
    nearest_distances: int      # actual values: 5.6724308700077755
```

`ParkingSchema` was also defined twice, the second silently replacing the first.

**Now:** a single validated load path in `data.load_dataset`, with explicit type
coercion and a clear error naming any missing column.

## 8. Simulated features were unseeded

`QueueLength`, `TrafficLevel`, `IsSpecialDay` and `VehicleTypeWeight` were drawn
with `np.random` at run time with no seed, so no two runs — and no two readers —
ever saw the same prices.

**Now:** the simulated columns are frozen into the committed dataset. The
pipeline draws no randomness at all; `test_pipeline_is_reproducible` asserts two
runs agree exactly.

## 9. Duplicate readings (found during this work)

The dataset holds 152 rows that share a `(lot, timestamp)` with another row.
`Occupancy` and `Capacity` agree within each group — these are genuine duplicate
readings in the source — but the *synthetic* `QueueLength` differs, because the
original script drew a fresh random value per row.

Left in place, these break the competitor join: the price for a timestamp
becomes an average over duplicates rather than the lot's own price.

**Now:** `data.load_dataset` collapses each group to its first reading, removing
86 rows and leaving a well-formed panel of 18,282. No real measurement is lost.
Covered by `test_dataset_has_one_row_per_lot_timestamp`.

## 10. The Bokeh "real-time" loop redrew the whole figure

```python
for i in range(len(df)):
    source.stream(new_data, rollover=100)
    show(p, notebook_handle=True)      # <- new plot every iteration
```

`show()` inside the loop emits a fresh plot per row instead of updating the
existing one; `push_notebook(handle=handle)` is the intended call, and `handle`
was captured but never used.

**Now:** figures are rendered headlessly to PNG by `plots.py`. The streaming
dashboard is out of scope for the core pipeline — see *Streaming* in the README.

---

## Not changed

* **The three-model structure.** Rule-based → demand-based → competition-aware
  is intact, as are the relative weights between demand drivers (1.0, 0.5, −0.7,
  1.5, 0.8) and the 500 m competitor radius.
* **The negative traffic weight.** `-0.7` on traffic means heavy traffic
  *lowers* price. That is arguable — congestion usually signals scarcity — but it
  is the submitted specification, so it stands. Change it in
  `config.Model2Config` if you want to test the alternative.
* **The dataset itself.** No rows were added, and none removed except the 86
  duplicates in §9.

## Known limitations

* The `20.0°N, 78.0°E` lot (`LOT_01`) is almost certainly a data-entry
  placeholder — it sits 1,560 km from the other thirteen, in central India,
  while the rest cluster in Guwahati. It is retained because dropping it would
  change the dataset; its isolation does mean Model 3 is a no-op for it.
* The currency unit is not recorded in the source data. The original mixed `$`
  in the matplotlib labels with `₹` in the Bokeh ones; figures here say "price"
  without a unit rather than guess.
* There is no ground truth for what the price *should* have been, so the models
  can be checked for internal consistency and plausibility but not scored for
  accuracy.
