# Development and Docker

## Development environment

Use Python 3.11, 3.12, or 3.13. Create an environment and install development dependencies:

```text
python -m venv .venv
```

On Windows:

```text
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe launch.py --check
```

On macOS/Linux:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python launch.py --check
```

Tests should run without API keys or network calls. Launcher tests use temporary project folders and mocked subprocesses, so they do not create environments or install packages. The CI workflow runs the unit/app tests and a real headless Streamlit health check on a Windows/macOS/Ubuntu × Python 3.11/3.12/3.13 matrix. Its health check verifies server startup; app tests verify rendered behavior. Check Actions for actual platform results rather than assuming a configured matrix has run.

Core packages and optional AI packages are separate. Do not import Transformers or PyTorch at app startup, and do not download model weights unless a user explicitly enables the optional model. Core package ranges allow compatible updates, so resolve and validate dependencies before a release; these files are not a full lockfile.

## Docker (optional)

Install [Docker Desktop](https://docs.docker.com/desktop/) or Docker Engine separately. From the project folder:

```text
docker build -t finsight .
docker run --rm -p 127.0.0.1:8501:8501 finsight
```

Open http://127.0.0.1:8501. Stop with Ctrl+C. The Docker image runs as an unprivileged user. The server listens on all interfaces **inside the container** so the published port works; the example publishes it only on the host's loopback interface.

For optional NewsAPI access:

```text
docker run --rm --env-file .env -p 127.0.0.1:8501:8501 finsight
```

The build excludes local secret files and virtual environments. Optional AI packages are excluded from the default image. Dependencies need internet during build; live data needs internet during use. Treat a deployment reachable by other people as a separate hosting decision that requires access controls, provider permissions, and an appropriate update process.

## Implementation references

- [Python `venv`](https://docs.python.org/3/library/venv.html): environment layout and direct interpreter use.
- [Python `subprocess`](https://docs.python.org/3/library/subprocess.html): argument arrays and process lifecycle.
- [Python on Windows](https://docs.python.org/3.13/using/windows.html): interpreter discovery and Windows launcher behavior.
- [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line), [configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml), and [Docker deployment](https://docs.streamlit.io/deploy/tutorials/docker).
