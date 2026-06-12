#!/usr/bin/env python3
"""ARIA — Autonomous Reflective Intelligence Architecture.

Thin entry point — the CLI lives in aria.cli so the installed package
(`pip install ariadx`) exposes the same `aria` command.
"""
from aria.cli import run

if __name__ == "__main__":
    run()
