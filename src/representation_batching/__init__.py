"""Representation-guided effective-batch scheduling utilities.

The package initializer deliberately avoids importing the PyTorch runtime so
that offline manifest generation can run in a lightweight NumPy environment.
"""
