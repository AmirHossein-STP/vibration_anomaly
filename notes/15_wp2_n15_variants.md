# 15. WP2: the 1500 rpm drop

MASTER_PLAN.md WP2, run 2026-10-15. H-a was answered in notes/14 §4: all
three 1500 rpm conditions are bad, so the cause is speed. This note tests
the mechanisms.

- Code:
  - `src/domain_adaptation/features/envelope.py`: opt-in variant arguments;
    the defaults are still v1.
  - `scripts/analysis/envelope_cache.py`: the `VARIANTS` registry.
  - `scripts/analysis/wp2_n15.py`: the pre-registered comparison.
  - `scripts/analysis/wp2_oracle.py`: the post-hoc diagnostic.
- Extraction: five caches, about 6.5 min in total on one core of the local
  Mac, run with the owner's approval.
- v1 was re-extracted after the code change. It matches the old cache to the
  CSV's 7 significant digits.
- Tables: `outputs/wp2/wp2_report.md`.

## 1. Pre-registered test: no variant explains the drop

The variants and the readings were written into `wp2_n15.py` before any
variant was extracted.

**Bar:** close at least half of the 0.315 recall gap (N15 mean recall
+0.16), without N09 falling by more than 0.05, with held-out FA within 0.03
of the 10% target.

| variant | hypothesis | N09 recall | N15 mean recall | verdict |
|---|---|---|---|---|
| v1 | reference | 0.740 | 0.425 | |
| `hb_revs15` (15-revolution windows) | H-b resolution | 0.743 | 0.427 | no (also N09 FA 0.151) |
| `hb_median` (peak / median) | H-b normalisation | 0.256 | 0.286 | no: degrades both speeds |
| `hc_reltol5` (±5% relative tolerance) | H-c, nominal speed | 0.767 | 0.239 | no: N15 worse |
| `hd_cpw_ses` (CPW + squared envelope, no band) | H-d | 0.142 | 0.185 | no: collapses everywhere |

By the pre-registered readings:
- H-b is rejected in both forms;
- H-c with nominal speed is rejected;
- H-d, in its CPW+SES form, is rejected *as a drop-in for the CWRU-trained
  model*.

Two side observations:
- Revolution windows rebalance the N15 conditions (F04 0.289 → 0.421, M01
  0.480 → 0.408) without moving their mean.
- The relative tolerance helps at 900 rpm and hurts at 1500 rpm.

## 2. Post-hoc diagnostic: the information is there, the CWRU boundary is not

This section is not pre-registered. It uses Paderborn labels to *diagnose*,
and nothing in it is chosen for the method.

**Measure:** window ROC-AUC for real damage vs healthy, under three fits.
The Paderborn fits use disjoint bearing units for fit and test, over 20
splits.

| features | condition | CWRU → PU | PU N09 → PU c | PU c → PU c (ceiling) |
|---|---|---|---|---|
| v1 | N09 | 0.890 | 0.935 | 0.935 |
| v1 | N15_M01_F10 | 0.736 | 0.642 | 0.782 |
| v1 | N15_M07_F04 | 0.611 | 0.559 | 0.703 |
| v1 | N15_M07_F10 | 0.758 | 0.633 | 0.787 |
| CPW+SES | N09 | **0.281** | 0.903 | 0.903 |
| CPW+SES | N15_M01_F10 | **0.393** | 0.661 | **0.892** |
| CPW+SES | N15_M07_F04 | **0.436** | 0.619 | **0.897** |
| CPW+SES | N15_M07_F10 | **0.412** | 0.641 | **0.883** |

How to read it:
- **v1 features carry less fault information at 1500 rpm.** A linear model
  fit on Paderborn itself reaches 0.70–0.79 at N15, against 0.935 at N09.
  The CWRU model is close to that ceiling. At 1500 rpm v1 mostly lacks the
  information, more than it fails to transfer.
- **The same rig at the other speed is no better than CWRU.** PU N09 → N15
  gives 0.56–0.64, so the difficulty is speed, not rig.
- **CPW+SES features do carry it** (0.88–0.90 at N15). That is weak
  support for H-d's premise: the fixed 1.5–5.5 kHz band discards fault
  information at 1500 rpm, and a label-free whitening recovers it.
- **But the CWRU-learned direction is inverted on CPW+SES** (AUC < 0.5).
  So H-d's fix cannot be bought with the current source-trained model.

### Which features turn

Single-feature AUC (fault vs healthy; 0.5 means no information, below 0.5
means the feature falls with damage):

| feature (v1) | CWRU | PU N09 | PU N15 (3 conditions) |
|---|---|---|---|
| `bpfo_h1` | 0.75 | 0.79 | 0.66–0.72 |
| `bpfi_h1` | 0.56 | 0.80 | 0.80–0.82 |
| `bpfi_sum` | 0.66 | 0.82 | 0.79–0.81 |
| `bpfi_sidebands` | 0.56 | 0.85 | **0.28–0.34** (inverted) |
| `env_kurtosis` | 0.20 | 0.11 | **0.47–0.50** (no information) |
| `env_crest` | 0.34 | 0.11 | **0.50–0.52** |
| `env_flatness` | **0.14** | **0.89** | 0.54–0.58 |

**Features keyed to fault orders keep their meaning everywhere.** These are
the BPFO/BPFI peaks: they rise with damage on CWRU and in all four PU
conditions.

**The aggregate features do not:**
- the envelope-spectrum shape statistics (kurtosis, crest, flatness);
- the shaft-rate sidebands.

These change sign or lose all information between rigs and between speeds:
- `env_flatness` points opposite ways on CWRU and on PU N09.
- Under CPW+SES, `env_kurtosis` and `env_crest` are 0.95 on CWRU and 0.21–0.35
  on PU. That is the inversion in the table above.

The 900 rpm success partly rests on features whose direction happened to
agree.

## 3. What this means for the plan

1. **N15 is not explained by a pre-registered mechanism.** H-b and H-c
   (nominal speed) are rejected. H-d is half-supported: the information is
   there under whitening, but it does not transfer.
   - The README must stop quoting N09 alone.
   - H-c with **measured** speed stays open. It needs the PU speed channel,
     to be re-extracted on Kaggle.
2. **A hypothesis for WP3, pre-registered here and not tested on Paderborn
   to avoid selection on target labels:**
   - A model restricted to geometry-keyed order peaks transfers more stably
     across speeds and rigs than one that also uses the envelope shape
     statistics and the sidebands.
   - It drops `env_*` and `bpfi_sidebands`, and keeps the BPFO/BPFI/FTF peaks.
   - Physics gives a reason in advance: only the peaks are defined by the
     bearing. The shape statistics describe the whole envelope spectrum,
     which belongs to the rig.
   - The test is WP3's stability selection across **source** datasets, then
     leave-one-dataset-out with XJTU-SY and Ottawa (WP5), and finally the IMS
     vault.
   - Its Paderborn number must not be quoted as evidence, because this note's
     diagnostic already looked at Paderborn.
3. **WP4 gains a sharper target.** Healthy-referenced scoring needs no
   source-learned direction at all: "how far each fault-order peak rose
   above this machine's healthy baseline". On CPW+SES features, that is the
   label-free way to use the information §2 shows is there without
   inheriting CWRU's inverted boundary. Recorded as a WP4 arm.
