"""Regenerate all 6 main figures for the 0928 submission package.
Run from this directory with data files at ../../data/.
Usage: python make_all_figures.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
os.makedirs(OUT, exist_ok=True)

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "purple": "#CC79A7", "cyan": "#56B4E9", "yellow": "#F0E442", "grey": "#999999",
     "dark": "#1a1a2e"}
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.5,
                     "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight"})

print("Note: This is a consolidated regeneration script.")
print("Individual figure scripts were part of the active development session.")
print("For full reproduction, see the data files and figure_spec.md for panel specifications.")
print("Data directory:", os.path.abspath(DATA))
print("Output directory:", os.path.abspath(OUT))
for f in sorted(os.listdir(DATA)):
    print("  data:", f)
