# Installation and troubleshooting

## Windows: first use

1. Install the 64 bit **Python 3.12** release from [python.org](https://www.python.org/downloads/windows/). If available, enable **Add Python to PATH**. Close and reopen Explorer or a terminal after installation.
2. Right click the downloaded Finsight ZIP, select **Extract All**, and choose a folder such as Documents. Open the extracted project folder.
3. Double-click **Start Finsight.bat**. The console shows package installation progress. First setup needs internet and can take several minutes; leave the console open.
4. The dashboard opens in your browser. Start with **Demo** data. Live data needs an internet connection; optional news uses your own key.
5. To stop, return to the console and press **Ctrl+C**. To start next time, double-click the same file.

The launcher tries the Windows Python launcher (`py`) first, preferring supported versions. It then falls back to `python`. It runs the virtual environment's Python directly, so PowerShell execution policy and activation scripts are unnecessary.

If Windows warns about a downloaded script, inspect the source and use your normal trusted download process. A corporate device may restrict scripts or package downloads; use your organization's supported installation process.

## macOS

Install Python 3.12 from [python.org](https://www.python.org/downloads/macos/), extract the ZIP, and open **Start Finsight.command**. If Finder blocks it or treats it as a text document, open Terminal in the extracted folder and run:

```sh
python3 launch.py
```

You can restore the executable permission with `chmod +x "Start Finsight.command"`. Python 3.11–3.13 are supported. The launcher prefers a versioned command such as `python3.12` when available.

## Linux

Install Python 3.11–3.13 and your distribution's `venv` support, then run `python3 launch.py` in the extracted folder. On Debian/Ubuntu, environment creation can require `python3-venv` or the matching versioned package. Use your distribution's package manager; Finsight does not modify system Python packages.

## Common problems

| Symptom | Action |
| --- | --- |
| Python is missing or unsupported | Install Python 3.12, then reopen the launcher. Python 3.14 is outside this project's current tested target. |
| Python opens Microsoft Store | Use the `py` command after installing Python from python.org, or correct Windows **App execution aliases** for Python. |
| An error mentions a missing `app.py` | Extract the entire ZIP. Keep its files together. |
| Package installation fails | Check internet, proxy settings, free disk space, and the exact pip error. Relaunch to retry. Do not run the project from a protected system folder. |
| `.venv` is broken after moving the folder or copying to another PC | Close Finsight and delete only the project's `.venv` folder. Relaunch so it is recreated for that machine and location. |
| Port 8501 is busy | Finsight may already be running: open http://127.0.0.1:8501. Otherwise run `python launch.py --port 8502`. On Windows you may use `py -3.12` in place of `python`. |
| Browser does not open | Paste the printed local URL into a browser. Keep the console open. |
| Live prices or fundamentals are missing | Check internet and retry later; upstream rate limits and coverage gaps are possible. The demonstration remains available. |
| News is unavailable | News is optional. Check your own provider key, subscription restrictions, and request quota. |
| FinBERT is unavailable | Read headlines without AI sentiment, or explicitly install optional packages with `python launch.py --install-ai`. |

## Updates

Close Finsight, then replace the project source with the new version. Do not copy `.venv` from someone else's computer. If requirements change, the launcher installs the new core requirements on the next start. To refresh packages manually, run `python launch.py --reinstall`.

Keep `.env`, `secrets.txt`, and `.streamlit/secrets.toml` private when sharing the project. Prefer the repository's clean ZIP over zipping your working environment.

## Setup diagnostics

Run `python launch.py --check` (Windows alternative: `py -3.12 launch.py --check`). This reports Python compatibility and missing project files without installing anything or opening the app. It does not verify internet access, provider availability, or a completed package installation. For installation errors, keep the console's actual error text for troubleshooting.
