# Pilot acceptance: what must hold before a unit is monitored

Run per unit after `ad bearing calibrate`. A unit that fails any **must** item is not
monitored. Fix the cause and recalibrate.

| # | Check | How | Must / should |
|---|---|---|---|
| 1 | Speed known for every recording | `calibrate` refuses otherwise | must |
| 2 | Bearing identified | `doctor` shows BPFO/BPFI for the unit; geometry-free is allowed but noted | should |
| 3 | Baseline ≥ 30 min, covering the unit's operating states | `calibrate` warns below 30 min; check the operating log | must |
| 4 | Baseline is self-consistent | the unit's `calibrate` line says **reads clean**: no healthy recording, judged against the rest of the unit's baseline, reaches the watch level | must |
| 5 | A later healthy recording reads healthy | record again a day later and run `infer`: verdict `healthy`, no `trend_alarm` | must |
| 6 | L0 severity plausible | `iso_zone` A or B on healthy recordings (if `iso20816` is configured) | should |
| 7 | Alarm handling agreed | who receives `watch`/`faulty`/`trend_alarm`, and how outcomes are logged | must |

If check 4 fails, the unit is in the most common failure mode seen in every study: its
"healthy" data contains an operating state, a mounting change or early damage that the
rest of the baseline does not (notes/14, 17, 24).
