#!/bin/sh
# Use the folder containing this file, even when Finder starts elsewhere.
PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
cd "$PROJECT_DIR" || exit 1

for PYTHON_COMMAND in python3.13 python3.12 python3.11 python3; do
    if command -v "$PYTHON_COMMAND" >/dev/null 2>&1; then
        "$PYTHON_COMMAND" "$PROJECT_DIR/launch.py" "$@"
        STATUS=$?
        if [ "$STATUS" -ne 0 ]; then
            printf '\nRead the error above. Setup help is in docs/INSTALL.md.\n'
            printf 'Press Return to close this window.'
            read -r REPLY
        fi
        exit "$STATUS"
    fi
done

MESSAGE='Python was not found. Install Python 3.12 from https://www.python.org/downloads/ and open Start Finsight.command again. See docs/INSTALL.md.'
printf '%s\n' "$MESSAGE"
if command -v osascript >/dev/null 2>&1; then
    osascript -e 'display dialog "Install Python 3.12 from python.org, then open Start Finsight.command again. See docs/INSTALL.md for help." with title "Finsight setup" buttons {"OK"} default button "OK"' >/dev/null 2>&1
fi
exit 1
