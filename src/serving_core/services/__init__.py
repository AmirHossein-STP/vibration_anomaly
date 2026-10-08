"""Service registry. Imports are lazy so the bearing service (numpy/scipy only)
does not need PyTorch installed, and vice versa."""
from __future__ import annotations

import importlib

# name -> (module, class)
SERVICES = {
    "bearing": ("serving_core.services.bearing", "BearingService"),
    "vibration": ("serving_core.services.vibration", "VibrationService"),
    "hydraulic": ("serving_core.services.hydraulic", "HydraulicService"),
}


def get_service(name: str):
    module, cls = SERVICES[name]
    return getattr(importlib.import_module(module), cls)()
