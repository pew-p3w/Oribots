"""Multibody experiment configs.

Each config names one single-robot base config (body, head, ball and feedback
settings) and one trained-weights source, plus the contest settings (K, hidden
size, arena, layouts, fitness weights, opponent mode, CMA-ES settings). A config
is a plain Python module, loaded and validated by ``multibody.config_loader``.

``_tiny.py`` is a seconds-long fixture for the test gate; ``spider_k4.py`` is the
first real experiment.
"""
