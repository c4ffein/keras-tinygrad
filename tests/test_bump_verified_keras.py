"""The version-bump logic keras-watch.yml runs — tested offline because its
sed predecessor silently merged adjacent strings into a nonsense version."""

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from bump_verified_keras import LOADER, bump  # noqa: E402


def test_appends_with_separating_comma():
    text = 'X = 1\nVERIFIED_KERAS_VERSIONS = ("3.15.0", "3.15.1")\nY = 2\n'
    out, new = bump(text, "3.16.0")
    assert new == ("3.15.0", "3.15.1", "3.16.0")
    parsed = ast.literal_eval("(" + re.search(r"VERIFIED_KERAS_VERSIONS = \(([^)]*)\)", out).group(1) + ")")
    assert parsed == new  # the sed bug produced ('3.15.0', '3.15.13.16.0')
    assert "3.15.13.16.0" not in str(parsed)


def test_idempotent():
    text = 'VERIFIED_KERAS_VERSIONS = ("3.15.1",)\n'
    out, new = bump(text, "3.15.1")
    assert out == text and new == ("3.15.1",)


def test_works_on_the_real_loader():
    text = pathlib.Path(LOADER).read_text()
    current = ast.literal_eval("(" + re.search(r"VERIFIED_KERAS_VERSIONS = \(([^)]*)\)", text).group(1) + ")")
    out, new = bump(text, "9.99.9")
    assert new == current + ("9.99.9",)  # appended to the loader's real tuple, nothing else touched
    compile(out, "<loader>", "exec")  # the rewritten file must stay valid python
