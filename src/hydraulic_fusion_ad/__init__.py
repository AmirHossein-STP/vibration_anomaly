"""Sensor-fusion anomaly detection on the ZeMA hydraulic systems dataset.

Sibling package to `vibration_ad` (CWRU/single-channel vibration MVP) --
see /HYDRAULIC_PLAN.md at the project root and notes/09_hydraulic_notes.md
for what's shared conceptually (healthy-only autoencoder, reconstruction
error as anomaly score) versus genuinely new here (multi-branch sensor
fusion across three native sampling rates, no windowing/spectrogram step).
"""
