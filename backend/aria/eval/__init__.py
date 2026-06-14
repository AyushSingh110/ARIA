"""Evaluation & labeling utilities (dataset loading, metrics, agreement).

This package backs the research-methodology scripts: two-annotator labeling,
Cohen's kappa, the frozen test-set split, and the metrics suite. It centralises
the one tricky concern — that each benchmark writes result files with slightly
different field names — behind a single normalised `Trace` view.
"""
