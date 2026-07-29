"""Vercel entry point.

Vercel's Python runtime looks for an ASGI callable named `app` in files under
api/. Everything real lives in the rolefit package so the same code runs under
uvicorn locally with no branching.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rolefit.api import app  # noqa: E402,F401
