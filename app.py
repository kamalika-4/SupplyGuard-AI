"""
SupplyGuard AI - Root Entry Point for Streamlit Community Cloud & Local Deployment
==================================================================================
Delegates execution to dashboard/app.py while ensuring proper module paths.
Runs seamlessly with both:
    streamlit run app.py
and
    streamlit run dashboard/app.py
"""
import os
import sys
import runpy

# Ensure repository root and key module paths are in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DASHBOARD_DIR = os.path.join(ROOT_DIR, "dashboard")
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)

# Target script path
TARGET_SCRIPT = os.path.join(DASHBOARD_DIR, "app.py")

if __name__ == "__main__":
    runpy.run_path(TARGET_SCRIPT, run_name="__main__")
