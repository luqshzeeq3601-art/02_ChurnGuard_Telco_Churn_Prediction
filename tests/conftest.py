"""Pytest configuration and global fixtures."""

import matplotlib

# Set non-interactive backend for headless test runs
matplotlib.use("Agg")
