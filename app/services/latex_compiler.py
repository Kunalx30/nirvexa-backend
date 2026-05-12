"""
app/services/latex_compiler.py
Compiles a LaTeX string to PDF bytes using Tectonic.

PLACEMENT: app/services/latex_compiler.py

HOW TECTONIC IS INSTALLED ON RENDER:
  Add this to your Render build command (one line):
  curl -L https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz | tar xz -C /usr/local/bin
  Tectonic auto-downloads required LaTeX packages on first compile.
  Subsequent compiles use the cache — takes ~2s.

LOCAL DEV (Windows/Mac):
  Install tectonic via: cargo install tectonic
  Or download binary from: https://github.com/tectonic-typesetting/tectonic/releases
  Add to PATH.
  On Windows if tectonic is not available, the function raises FileNotFoundError
  with a clear message so you can handle it in the route.
"""
import os
import logging
import subprocess
import tempfile
import sys

logger = logging.getLogger(__name__)

# Allow override via env var for non-standard install paths
if sys.platform == "win32":
    TECTONIC_BIN = os.path.join(os.getcwd(), "tectonic.exe")
else:
    TECTONIC_BIN = os.environ.get("TECTONIC_BIN", "tectonic")


def compile_latex(latex_code: str) -> bytes:
    """
    Compile a LaTeX string to PDF and return raw PDF bytes.

    Args:
        latex_code: Complete .tex document as a string.

    Returns:
        PDF file as bytes.

    Raises:
        FileNotFoundError: Tectonic binary not found on PATH.
        RuntimeError:      LaTeX compilation failed (stderr included in message).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "resume.tex")
        pdf_path = os.path.join(tmpdir, "resume.pdf")

        # Write .tex file
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_code)

        # Run tectonic
        # --keep-logs        → don't delete logs on success (helps debugging)
        # --keep-intermediates → keep .aux etc for multi-pass
        # Warnings like "Underfull \hbox" are NOT errors — we check for
        # actual errors by looking for "error:" in stderr, not just returncode.
        try:
            result = subprocess.run(
                [
                    TECTONIC_BIN,
                    "-X", "compile",
                    "--outdir", tmpdir,
                    "--keep-logs",
                    "--keep-intermediates",
                    tex_path,
                ],
                capture_output=True,
                timeout=60,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Tectonic binary not found. "
                "Install via: cargo install tectonic  "
                "or download from https://github.com/tectonic-typesetting/tectonic/releases"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("LaTeX compilation timed out after 60 seconds.")

        stderr = result.stderr.decode("utf-8", errors="replace")
        stdout = result.stdout.decode("utf-8", errors="replace")

        # PDF produced = success, even if returncode != 0 due to warnings
        if os.path.exists(pdf_path):
            if result.returncode != 0:
                logger.warning(
                    f"[LaTeX] Tectonic returned code {result.returncode} but PDF exists — "
                    f"treating as success (likely only warnings).\nSTDERR: {stderr[:400]}"
                )
            with open(pdf_path, "rb") as f:
                return f.read()

        # No PDF produced — real failure
        logger.error(f"[LaTeX] Tectonic failed:\nSTDERR: {stderr}\nSTDOUT: {stdout}")

        # Extract only actual error lines for cleaner error message
        error_lines = [
            line for line in stderr.splitlines()
            if line.strip().startswith("error:")
        ]
        error_summary = "\n".join(error_lines) if error_lines else stderr[:800]
        raise RuntimeError(f"LaTeX compilation failed:\n{error_summary}")