import os

import pytest

from criterion_lens.utils.env import load_env


def test_load_env_parses_values_without_overriding_existing_variables(
    tmp_path, monkeypatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        '# local settings\n'
        'CRITERION_LENS_TEST_EXISTING=file-value\n'
        'CRITERION_LENS_TEST_QUOTED="value with spaces"\n'
        'CRITERION_LENS_TEST_EQUALS=first=second\n'
        'CRITERION_LENS_TEST_HASH=value#suffix\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CRITERION_LENS_TEST_EXISTING", "process-value")

    loaded = load_env(env_path)

    assert loaded == 3
    assert os.environ["CRITERION_LENS_TEST_EXISTING"] == "process-value"
    assert os.environ["CRITERION_LENS_TEST_QUOTED"] == "value with spaces"
    assert os.environ["CRITERION_LENS_TEST_EQUALS"] == "first=second"
    assert os.environ["CRITERION_LENS_TEST_HASH"] == "value#suffix"


def test_load_env_reports_malformed_lines_without_echoing_values(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("invalid secret-value\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_env(env_path)

    assert "secret-value" not in str(error.value)


def test_load_env_reports_missing_explicit_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_env(tmp_path / "missing.env")
