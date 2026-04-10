from __future__ import annotations

import subprocess
import sys
import tempfile


def test_great_expectations_import_is_warning_clean():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import warnings; "
                    "warnings.simplefilter('error'); "
                    "import great_expectations"
                ),
            ],
            check=False,
            capture_output=True,
            cwd=temp_dir,
            text=True,
        )

    assert result.returncode == 0, result.stderr or result.stdout
