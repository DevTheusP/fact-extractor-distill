import ast
import json
import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "fact_extractor_distill.ipynb"


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NB, as_version=4)


def test_notebook_is_valid(nb):
    nbformat.validate(nb)
    json.loads(NB.read_text())


def test_notebook_is_in_sync_with_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "notebooks" / "build_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert NB.read_text() == mod.render(), "rode python notebooks/build_notebook.py"


def test_code_cells_parse_as_python(nb):
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)


def test_config_cell_on_top(nb):
    first_code = next(c for c in nb.cells if c.cell_type == "code")
    for name in ("SMOKE_TEST", "STUDENT_MODEL", "N_TRAIN", "EPOCHS", "HF_REPO_MODEL", "PUSH_TO_HUB"):
        assert re.search(rf"^{name}\b", first_code.source, re.M), name
    assert re.search(r"^PUSH_TO_HUB = False", first_code.source, re.M)


def test_no_hardcoded_tokens(nb):
    src = "\n".join(c.source for c in nb.cells)
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", src)
    assert 'UserSecretsClient().get_secret("HF_TOKEN")' in src
    assert re.search(r'os\.environ\["HF_TOKEN"\]\s*=\s*UserSecretsClient', src)
    # o token nunca é impresso nem gravado
    assert not re.search(r"print\([^)]*HF_TOKEN", src)


def test_every_cli_module_called_exists(nb):
    src = "\n".join(c.source for c in nb.cells)
    for mod in set(re.findall(r'"-m", "(distill\.\w+)"', src)):
        assert (ROOT / (mod.replace(".", "/") + ".py")).exists(), mod
