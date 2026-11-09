import os
import subprocess

import pytest

from criterion_lens.utils.env import load_env
from scripts.kaggle import kaggle_env


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


def test_load_env_overrides_only_explicitly_selected_keys(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KAGGLE_API_TOKEN=file-token\n"
        "KAGGLE_USERNAME=file-user\n"
        "CRITERION_LENS_TEST_PRESERVED=file-value\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_API_TOKEN", "process-token")
    monkeypatch.setenv("KAGGLE_USERNAME", "process-user")
    monkeypatch.setenv("CRITERION_LENS_TEST_PRESERVED", "process-value")

    load_env(env_path, override_keys={"KAGGLE_API_TOKEN", "KAGGLE_USERNAME"})

    assert os.environ["KAGGLE_API_TOKEN"] == "file-token"
    assert os.environ["KAGGLE_USERNAME"] == "file-user"
    assert os.environ["CRITERION_LENS_TEST_PRESERVED"] == "process-value"


def test_kaggle_cli_uses_dotenv_credentials_without_echoing_token(
    tmp_path, monkeypatch, capsys
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KAGGLE_API_TOKEN=dotenv-token\nKAGGLE_USERNAME=syedashhermajid\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_API_TOKEN", "stale-process-token")
    monkeypatch.setenv("KAGGLE_USERNAME", "stale-process-user")
    monkeypatch.setattr(
        kaggle_env,
        "load_env",
        lambda *, override_keys: load_env(env_path, override_keys=override_keys),
    )
    monkeypatch.setattr(kaggle_env, "kaggle_executable", lambda: "kaggle")
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="kernel list", stderr="")

    monkeypatch.setattr(kaggle_env.subprocess, "run", fake_run)

    result = kaggle_env.run_kaggle(["kernels", "list", "--mine"], capture_output=True)

    assert observed["environment"]["KAGGLE_API_TOKEN"] == "dotenv-token"
    assert observed["environment"]["KAGGLE_USERNAME"] == "syedashhermajid"
    assert result.stdout == "kernel list"
    assert "dotenv-token" not in capsys.readouterr().out
