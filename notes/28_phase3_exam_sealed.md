# 28. Phase 3: the new confirmation sets, sealed (IMS is spent)

Written 2026-10-22, at the start of Phase 3 (MASTER_PLAN Phase 3). IMS was the final exam for
method v1.1 (notes/27) and is now **development data**. Whatever Phase 3 freezes needs a
confirmation set nobody has seen. Two published run-to-failure sets were found that the box
can reach (Mendeley):

| registry name | what (from the Mendeley landing pages only) | size at the publisher |
|---|---|---|
| `ferrara` | Univ. of Ferrara: self-aligning double-row ball bearings, 6 accelerated run-to-failure tests (E1–E6), constant speed (inverter-driven), radial load from a load cell, one radial accelerometer. Gabrielli et al., Data in Brief 55, 110620 (2024). Two records: `htk59pp5wx` v2 (E1–E3) and `zz8hpyx939` v2 (E4–E6). CC BY-NC 4.0 | 12.05 GB (6 × .7z + 2 readme PDFs) |
| `kaist_rtf` | KAIST: one NSK 6205 ball bearing run to failure in 128 h, vibration and temperature, 25.6 kHz, hourly files. Mendeley `5hcdd3tdvb` v6. CC BY 4.0 | 4.33 GB (1 zip) |

## 1. How they are sealed
Together they would leave the vault with almost no room (16.6 GB free), so they are **pinned
by checksum** instead of downloaded (`stage: pin`, new in `remote/datasets/fetch.py`):
- `fetch` recorded each file's name, size and the publisher's sha256 and downloaded nothing.
  The manifests are committed in `remote/datasets/manifests/{ferrara,kaist_rtf}.json`.
- `unseal` (allowed only with `--confirm P3-EXAM`) downloads them onto the SSD and refuses any
  file whose sha256 differs from the pinned one. Any change the publisher made after today
  therefore shows up at the exam.
- Nobody has looked at a signal, a label, a file list inside an archive, or a readme. The two
  readme PDFs are inside the seal too.

## 2. What stays open until the exam protocol is written
The published articles (Data in Brief) were not readable from here (403 at DOAJ and
ScienceDirect, 2026-10-22). Before unsealing, the protocol (units, speed, geometry, baseline
length, documented failures, pre-registered readings) is written from those **articles only**,
as notes/27 §1–3 was for IMS. It is written after Phase 3 freezes, because the readings depend
on what is frozen (a detection score, a trigger, or both), and in any case before `unseal`.

Known risks, stated now:
- the Ferrara bearing is self-aligning double-row (like IMS's double-row Rexnord, where only
  geometry-free detection worked);
- KAIST is one bearing (n = 1), so it can confirm or refute only a single case.

## 3. Unsealing record
(empty)
