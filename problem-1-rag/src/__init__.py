# Cost-Efficient RAG Package
import sys
import os
from pathlib import Path

if sys.platform == "win32":
    for candidate in [
        r"C:\Program Files\Common Files\microsoft shared\ClickToRun",
        str(Path(__file__).parent.parent / ".venv" / "Scripts"),
    ]:
        if os.path.isdir(candidate):
            try:
                os.add_dll_directory(candidate)
            except Exception:
                pass
