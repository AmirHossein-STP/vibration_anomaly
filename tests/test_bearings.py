"""Every catalog bearing reproduces its published fault frequencies (notes/23 §1)."""
from __future__ import annotations

import pytest

from domain_adaptation.bearings import CATALOG, FaultOrders, bsf_of, from_dims, from_factors, lookup

PUBLISHED_ORDERS = {          # designation: (BPFO, BPFI, tolerance) in orders
    "SKF 6205-2RS JEM": (3.5848, 5.4152, 0.001),
    "6203 (Paderborn KAt)": (3.053, 4.947, 0.001),
    "ER16K": (3.57, 5.43, 0.01),
    "LDK UER204": (107.91 / 35, None, 0.002),
    "MFPT test bearing": (81.125 / 25, 118.875 / 25, 1e-9),
    "Rexnord ZA-2115": (236.4 / (2000 / 60), 296.9 / (2000 / 60), 0.01),
}


@pytest.mark.parametrize("name", sorted(PUBLISHED_ORDERS))
def test_catalog_reproduces_published_fault_orders(name):
    g = lookup(name)
    bpfo, bpfi, tol = PUBLISHED_ORDERS[name]
    assert abs(g.bpfo - bpfo) <= tol
    if bpfi is not None:
        assert abs(g.bpfi - bpfi) <= tol


def test_every_entry_is_sane_and_cited():
    for name, (g, source) in CATALOG.items():
        assert source and 1.0 < g.bpfo < g.bpfi < 20.0, name
        if not isinstance(g, FaultOrders):
            assert abs((g.bpfo + g.bpfi) - g.n_balls) < 1e-9     # BPFO + BPFI = number of balls


def test_mfpt_bsf_matches_shipped_value_within_rounding():
    g = from_dims(8, 0.235, 1.245)
    assert abs(bsf_of(g) * 25 - 63.91) < 0.1


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        from_dims(8, 30.0, 28.5)
    with pytest.raises(ValueError):
        from_factors(bpfo=5.0, bpfi=3.0)
    with pytest.raises(KeyError):
        lookup("no such bearing")
