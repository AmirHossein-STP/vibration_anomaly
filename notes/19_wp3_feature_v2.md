# 19. WP3: feature set v2 for the physics score

MASTER_PLAN.md WP3. Written 2026-10-22. **§1–§3 were written before any v2 number was computed.**
It follows from notes/17 §5.3: ball and cage faults are not covered by the six race
peaks (Ottawa 2018 ball 0.13, Ottawa 2023 ball 0.02, cage 0.07), and the
order-tracked outer-race line is weak on Ottawa 2018 (0.54).

## 1. What changes, and what stays

The pipeline is unchanged up to the envelope spectrum: 12 kHz, 1 s windows (20 revolutions
order-tracked on Ottawa 2018), CPW + squared envelope. The healthy-referenced z-score and
cross-fitted calibration (notes/17 §5.2) also stay as they are. `envelope.py` stays frozen. v2
lives in `features/physics_v2.py`. Peaks read (each divided by the envelope sum over
0.5–30 orders):

| family | lines |
|---|---|
| outer | BPFO × 1–5 |
| inner | BPFI × 1–5, BPFI ± 1 order (shaft-rate sidebands) |
| ball | BSF × 1–3 (BSF = D/2d · (1 − (d/D · cos α)²)) |
| cage | FTF × 1–2 |

Tolerance is ±max(0.08, 0.02 × order). The 2% covers slip and speed error, which grow with the
harmonic number (Shen et al. ±5%, Xie et al. ±0.1 order). At the harmonics v1 used, this is the
v1 tolerance until order 4.

## 2. Arms (all cross-fitted, conformal α = 0.10, folds as in notes/17)

- `d_xf`: the frozen arm d, from the frozen caches. **This is the comparison.**
- `f1`: family-mean aggregation over the v1 lines only (BPFO and BPFI × 1–3, new
  tolerance). It isolates the aggregation effect.
- **`v2fam` (primary v2)**: per family, the mean z of its lines. Score = max over the 4 families.
  Averaging within a family is the guard against the multiplicity of 17 lines.
- `v2max`: max z over all 17 lines (the naive extension).
- Typing v2: the family with the largest family z → outer / inner / ball / cage.

## 3. Gate (pre-registered)

**v2fam replaces v1** if both hold:
- (a) race-fault AUC ≥ `d_xf` − 0.02 on **every** dataset: CWRU, Paderborn ×4,
  Ottawa 2018, Ottawa 2023 machine and own, MFPT;
- (b) ball+cage AUC improves by ≥ 0.10 over `d_xf` on ≥ 2 of: Ottawa 2018 (ball),
  Ottawa 2023 machine, Ottawa 2023 own.

Otherwise **v1 stays** (MASTER_PLAN §7). The finding is still reported, and a family that helps
without hurting is noted for WP9 as a diagnosis-only output.

Also reported:
- whether `f1` alone changes anything, which separates aggregation from new lines;
- typing on ball/cage faults;
- XJTU-SY: end-of-life typing of the cage failures 1_4 and 2_3 (documented cage), a qualitative check.

Every one of these datasets has now been looked at by the project (notes/16–18). So this is a
**development** comparison, not a confirmation. Confirmation of whatever is frozen comes
only from IMS (WP11).

---

## 4. Results (after computing, 2026-10-22)

Runs: `remote/box run v2cache` (all six datasets, ~30 s) and `remote/box run wp3score`.
Table: `outputs/wp3/wp3_report.md`.

**Gate: v2fam NOT adopted, v1 stays.**
- (a) **fails**. Race AUC drops by more than 0.02 against frozen arm d on 5 of 10 rows:
  - Paderborn 1500 rpm: 0.66–0.75 vs 0.84–0.86;
  - Ottawa 2018: 0.77 vs 0.92;
  - MFPT: 0.95 vs 1.00.
- (b) passes on Ottawa 2023: ball+cage AUC +0.14 (machine baseline) and +0.13 (own).
  It fails on Ottawa 2018 ball (−0.06).

| arm | what it shows |
|---|---|
| `f1` (aggregation only) | Averaging the v1 lines within a race is slightly *worse* than taking their max on 8 of 10 rows. The max of single-line z is the better detector. A fault often lights one harmonic, not all three |
| `v2fam` | Averaging 5 harmonics dilutes the race lines more. The ball/cage families do add detection on Ottawa 2023 (race AUC 0.94 machine, ball recall 0.51), but they *steal typing*: race typing falls to 0.34–0.52 because ball/cage lines rise on race faults too |
| `v2max` | The naive max over 17 lines is the best arm on Ottawa 2018 (0.944), CWRU (0.947) and Ottawa 2023 machine (0.856 vs 0.712). But on Paderborn 1500 rpm it loses up to 0.05 and its FA rises (0.16–0.24): more lines means more chances for a healthy line to spike |

What this means:
- The six v1 lines with max-z remain the most robust single choice. Every richer
  variant wins somewhere and loses somewhere. That is the same pattern as WP2 (notes/15)
  and as kurtosis (notes/18 §6), and it is why the frozen method stays frozen.
- **Ball and cage faults remain uncovered** by a robust detector. Their envelope lines
  are weak and they disturb the race lines. They stay a documented gap (MASTER_PLAN §7).
  WP6's synthetic-fault work is the next place to look.
- The fleet-baseline problem on Ottawa 2023 is partly a *feature* problem, not only a baseline
  one: richer line sets raise its AUC from 0.71 to 0.86 (v2max) or 0.94 (v2fam), and v2fam
  brings its worst-unit FA from 0.89 down to 0.32.
  This is noted for WP9: when no own-unit history exists yet (a new machine), a richer line
  set may help the fleet baseline. Not adopted, and not tested on a second dataset.
