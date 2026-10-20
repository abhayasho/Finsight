#!/usr/bin/env python3
"""Set up a local environment and launch Finsight without shell activation.

Only the standard library is imported, so a fresh Python installation can run
this file. Importing this module never installs anything or starts the app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_MINORS = (11, 12, 13)
STAMP_NAME = ".finsight-core.json"


class LauncherError(Exception):
    """An expected setup problem with an actionable message."""


def python_supported(version: tuple[int, ...]) -> bool:
    return version[0] == 3 and version[1] in SUPPORTED_MINORS


def environment_python(root: Path, platform: str | None = None) -> Path:
    """Return the environment interpreter, including paths containing spaces."""
    windows = (platform or os.name) == "nt"
    return root / ".venv" / ("Scripts/python.exe" if windows else "bin/python")


def requirements_fingerprint(requirements: Path, runtime: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"finsight-core-v1\0")
    digest.update(runtime.encode("utf-8"))
    digest.update(b"\0")
    digest.update(requirements.read_bytes())
    return digest.hexdigest()


def run_checked(command: list[str], root: Path, action: str) -> None:
    """Run argument arrays directly; never invoke a shell or activation script."""
    try:
        subprocess.run(command, cwd=str(root), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LauncherError(
            f"{action} failed. Check the error above and your internet connection. "
            "Then run the launcher again. On Linux, environment creation may "
            "require the python3-venv package."
        ) from exc


def inspect_runtime(python: Path, root: Path) -> str:
    script = (
        "import json, platform, sys; "
        "print(json.dumps({'version': list(sys.version_info[:2]), "
        "'identity': platform.python_version() + '|' + sys.platform + '|' "
        "+ platform.machine()}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script], cwd=str(root), check=True,
            capture_output=True, text=True, timeout=30,
        )
        runtime = json.loads(result.stdout)
        if not python_supported(tuple(runtime["version"])):
            raise ValueError("unsupported environment Python")
        return str(runtime["identity"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise LauncherError(
            "The .venv environment cannot run with a supported Python. "
            "Close Finsight, remove only the .venv folder in this extracted "
            "project, and start again with Python 3.11, 3.12, or 3.13. "
            "Environments cannot be copied between computers or moved folders."
        ) from exc


def read_stamp(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def prepare_environment(root: Path, reinstall: bool = False) -> Path:
    python = environment_python(root)
    if not python.exists():
        if (root / ".venv").exists():
            raise LauncherError(
                "A partial .venv folder already exists. Remove only that folder "
                "from this project and launch again to recreate it."
            )
        print("First launch: creating Finsight's private Python environment...", flush=True)
        run_checked([sys.executable, "-m", "venv", str(root / ".venv")], root,
                    "Creating the Python environment")

    runtime = inspect_runtime(python, root)
    fingerprint = requirements_fingerprint(root / "requirements.txt", runtime)
    stamp_path = root / ".venv" / STAMP_NAME
    if reinstall or read_stamp(stamp_path).get("fingerprint") != fingerprint:
        print("Installing core packages. The first installation may take several minutes...", flush=True)
        run_checked(
            [str(python), "-m", "pip", "install", "--upgrade", "-r", str(root / "requirements.txt")],
            root, "Installing core packages",
        )
        # Write only after pip succeeds, so interrupted setup is retried.
        stamp_path.write_text(json.dumps({"fingerprint": fingerprint, "runtime": runtime}),
                              encoding="utf-8")
    return python


def streamlit_command(python: Path, root: Path, port: int) -> list[str]:
    return [
        str(python), "-m", "streamlit", "run", str(root / "app.py"),
        "--server.address=127.0.0.1", f"--server.port={port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
    ]


def check_port(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", port))
    except OSError as exc:
        raise LauncherError(
            f"Port {port} is already in use or unavailable. Finsight may already "
            f"be open at http://127.0.0.1:{port}. Otherwise try: "
            "python launch.py --port 8502"
        ) from exc


def open_browser_when_ready(url: str) -> None:
    """Wait briefly for the local server before opening the default browser."""
    for _ in range(120):
        try:
            with urllib.request.urlopen(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up and launch Finsight locally.")
    parser.add_argument("--check", action="store_true", help="Check Python/project without installing or starting.")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser.")
    parser.add_argument("--port", type=int, default=8501, help="Local server port (default: 8501).")
    parser.add_argument("--reinstall", action="store_true", help="Resolve/install core requirements again.")
    parser.add_argument("--install-ai", action="store_true", help="Install optional local AI packages, then exit.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not python_supported(sys.version_info):
            raise LauncherError(
                f"This launcher is running Python {sys.version_info.major}.{sys.version_info.minor}. "
                "Install Python 3.12 from https://www.python.org/downloads/ "
                "(supported: 3.11–3.13), then run the launcher again."
            )
        required = ["app.py", "requirements.txt"]
        if args.install_ai:
            required.append("requirements-ai.txt")
        missing = [name for name in required if not (PROJECT_ROOT / name).is_file()]
        if missing:
            raise LauncherError(
                f"Missing project files: {', '.join(missing)}. "
                "Extract the entire Finsight ZIP into a folder before launching."
            )
        if args.check:
            print(f"Python {sys.version.split()[0]}: supported")
            print(f"Project: {PROJECT_ROOT}")
            print("Required project files: present. No packages installed and no server started.")
            return 0

        python = prepare_environment(PROJECT_ROOT, args.reinstall)
        if args.install_ai:
            print("Installing optional AI packages. Model weights download only when selected in the app.", flush=True)
            run_checked(
                [str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements-ai.txt")],
                PROJECT_ROOT, "Installing optional AI packages",
            )
            print("Optional AI packages installed. Start Finsight and select FinBERT to use them.")
            return 0

        check_port(args.port)
        url = f"http://127.0.0.1:{args.port}"
        print(f"\nFinsight is starting at {url}\nKeep this window open. Press Ctrl+C here to stop.\n", flush=True)
        if not args.no_browser:
            threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()
        result = subprocess.run(streamlit_command(python, PROJECT_ROOT, args.port), cwd=str(PROJECT_ROOT))
        if result.returncode:
            raise LauncherError(
                "Finsight stopped with an error. Read the message above. "
                "For dependency problems, try: python launch.py --reinstall"
            )
        return 0
    except KeyboardInterrupt:
        print("\nFinsight stopped.")
        return 0
    except (LauncherError, OSError) as exc:
        print(f"\nFinsight could not start: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
