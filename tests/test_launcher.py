"""Exercise launcher behavior without running pip, a server, or a browser."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location("finsight_launcher", Path(__file__).parents[1] / "launch.py")
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="Finsight test with spaces ")
        self.root = Path(self.temporary.name)
        (self.root / "requirements.txt").write_text("streamlit>=1.42,<2\n", encoding="utf-8")
        (self.root / "app.py").write_text("# test app\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def existing_environment(self):
        python = launch.environment_python(self.root)
        python.parent.mkdir(parents=True)
        python.touch()
        return python

    def test_python_support_is_explicit(self):
        self.assertTrue(launch.python_supported((3, 11, 0)))
        self.assertTrue(launch.python_supported((3, 13, 0)))
        self.assertFalse(launch.python_supported((3, 10, 0)))
        self.assertFalse(launch.python_supported((3, 14, 0)))

    def test_interpreter_paths_for_windows_and_posix(self):
        self.assertEqual(launch.environment_python(self.root, "nt"), self.root / ".venv" / "Scripts" / "python.exe")
        self.assertEqual(launch.environment_python(self.root, "posix"), self.root / ".venv" / "bin" / "python")

    def test_fingerprint_changes_with_requirements_and_runtime(self):
        requirements = self.root / "requirements.txt"
        original = launch.requirements_fingerprint(requirements, "3.12|win32|AMD64")
        self.assertNotEqual(original, launch.requirements_fingerprint(requirements, "3.13|win32|AMD64"))
        requirements.write_text("streamlit>=1.43,<2\n", encoding="utf-8")
        self.assertNotEqual(original, launch.requirements_fingerprint(requirements, "3.12|win32|AMD64"))

    def test_invalid_stamp_triggers_safe_empty_value(self):
        stamp = self.root / "stamp.json"
        self.assertEqual(launch.read_stamp(stamp), {})
        for contents in ("not json", "[]", "null"):
            stamp.write_text(contents, encoding="utf-8")
            self.assertEqual(launch.read_stamp(stamp), {})

    def test_first_setup_uses_argument_arrays_and_writes_stamp_after_success(self):
        python = launch.environment_python(self.root)

        def fake_run(command, root, action):
            if command[1:3] == ["-m", "venv"]:
                python.parent.mkdir(parents=True)
                python.touch()

        with mock.patch.object(launch, "run_checked", side_effect=fake_run) as run, \
                mock.patch.object(launch, "inspect_runtime", return_value="3.12.1|test|test"):
            self.assertEqual(launch.prepare_environment(self.root), python)
        creation, install = run.call_args_list
        self.assertEqual(creation.args[0][1:3], ["-m", "venv"])
        self.assertEqual(creation.args[0][-1], str(self.root / ".venv"))
        self.assertEqual(install.args[0], [str(python), "-m", "pip", "install", "--upgrade", "-r", str(self.root / "requirements.txt")])
        self.assertIn("fingerprint", launch.read_stamp(self.root / ".venv" / launch.STAMP_NAME))

    def test_unchanged_setup_does_not_install(self):
        python = self.existing_environment()
        runtime = "3.12.1|test|test"
        stamp = self.root / ".venv" / launch.STAMP_NAME
        stamp.write_text(json.dumps({"fingerprint": launch.requirements_fingerprint(self.root / "requirements.txt", runtime)}), encoding="utf-8")
        with mock.patch.object(launch, "run_checked") as run, \
                mock.patch.object(launch, "inspect_runtime", return_value=runtime):
            self.assertEqual(launch.prepare_environment(self.root), python)
        run.assert_not_called()

    def test_failed_install_is_not_marked_as_completed(self):
        self.existing_environment()
        with mock.patch.object(launch, "run_checked", side_effect=launch.LauncherError("installation failed")), \
                mock.patch.object(launch, "inspect_runtime", return_value="3.12.1|test|test"):
            with self.assertRaises(launch.LauncherError):
                launch.prepare_environment(self.root)
        self.assertFalse((self.root / ".venv" / launch.STAMP_NAME).exists())

    def test_partial_environment_is_not_deleted_automatically(self):
        (self.root / ".venv").mkdir()
        marker = self.root / ".venv" / "important.txt"
        marker.write_text("keep", encoding="utf-8")
        with mock.patch.object(launch, "run_checked") as run:
            with self.assertRaisesRegex(launch.LauncherError, "partial"):
                launch.prepare_environment(self.root)
        run.assert_not_called()
        self.assertTrue(marker.exists())

    def test_streamlit_is_loopback_and_spaced_paths_stay_one_argument(self):
        python = launch.environment_python(self.root)
        command = launch.streamlit_command(python, self.root, 8502)
        self.assertEqual(command[0], str(python))
        self.assertEqual(command[4], str(self.root / "app.py"))
        self.assertIn("--server.address=127.0.0.1", command)
        self.assertIn("--server.port=8502", command)

    def test_subprocess_does_not_use_shell(self):
        command = ["Python path with spaces", "-m", "pip", "install", "a;echo unsafe"]
        with mock.patch.object(launch.subprocess, "run") as run:
            launch.run_checked(command, self.root, "Test")
        run.assert_called_once_with(command, cwd=str(self.root), check=True)

    def test_runtime_validation_reports_a_broken_environment(self):
        with mock.patch.object(launch.subprocess, "run", side_effect=OSError("missing executable")):
            with self.assertRaisesRegex(launch.LauncherError, "cannot be copied"):
                launch.inspect_runtime(self.root / "missing python", self.root)

    def test_runtime_accepts_supported_environment(self):
        completed = subprocess.CompletedProcess([], 0, json.dumps({"version": [3, 12], "identity": "3.12.1|win32|AMD64"}))
        with mock.patch.object(launch.subprocess, "run", return_value=completed):
            self.assertEqual(launch.inspect_runtime(self.root / "python", self.root), "3.12.1|win32|AMD64")

    def test_check_has_no_install_server_or_browser_side_effects(self):
        with mock.patch.object(launch, "PROJECT_ROOT", self.root), \
                mock.patch.object(launch, "python_supported", return_value=True), \
                mock.patch.object(launch, "prepare_environment") as prepare, \
                mock.patch.object(launch.subprocess, "run") as run, \
                mock.patch.object(launch.webbrowser, "open") as browser, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.main(["--check"]), 0)
        self.assertIn("No packages installed", output.getvalue())
        prepare.assert_not_called()
        run.assert_not_called()
        browser.assert_not_called()

    def test_check_missing_files_has_actionable_error(self):
        (self.root / "app.py").unlink()
        with mock.patch.object(launch, "PROJECT_ROOT", self.root), \
                mock.patch.object(launch, "python_supported", return_value=True), \
                contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(launch.main(["--check"]), 1)
        self.assertIn("Extract the entire", output.getvalue())

    def test_rejects_invalid_port(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for port in ("0", "65536", "-1"):
                with self.assertRaises(SystemExit):
                    launch.parse_args(["--port", port])


if __name__ == "__main__":
    unittest.main()
