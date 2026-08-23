"""Skeleton for a future optimized regular analytic 3PCF route."""
from __future__ import annotations


class AnalyticThreePCFCalculator:
    is_operational = False

    def __init__(self, terms, *, config, **kwargs):
        self.terms = tuple(terms)
        self.config = config

    def compute_zetak(self, *args, **kwargs):
        raise NotImplementedError("analytic 3PCF route is not implemented")
