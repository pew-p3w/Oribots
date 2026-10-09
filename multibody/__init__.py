"""Multibody ball contest: K trained robots compete for one ball in a walled arena.

This package is a self-contained layer on top of the single-robot Oribots setup.
It never edits the single-robot code (``src/``, ``brains/``, ``config/``,
``engine/``, ``run.py``, ``jobs/``, ``hpc/``); it imports from it. Every module
name here is deliberately distinct from the single-robot module names, so
nothing is shadowed when both are on ``sys.path``.

The design and the full requirement set live in
``.kiro/specs/multibody-ball-contest/`` (requirements.md, design.md, tasks.md).
Run its checks with ``make -C multibody check``.
"""
