# Code execution tool. The LLM never runs code itself; the assistant calls
# this tool, which executes the source in a separate python process and
# returns captured stdout and stderr.

import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 30


def run_code(source):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(source)
        script = Path(f.name)
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "", f"Execution stopped: took longer than {TIMEOUT_SECONDS} seconds."
    finally:
        script.unlink(missing_ok=True)
    return result.stdout, result.stderr
