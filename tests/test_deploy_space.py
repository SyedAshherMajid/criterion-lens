import subprocess
import sys

from scripts.deploy_space import PROJECT_ROOT, prepare_space_payload


def test_space_payload_is_secret_free_and_importable(tmp_path) -> None:
    payload = prepare_space_payload(tmp_path / "payload")

    assert (payload / "README.md").is_file()
    assert (payload / "requirements.txt").is_file()
    assert (payload / "app.py").is_file()
    assert (payload / "app" / "ui.py").is_file()
    assert (payload / "app" / "examples" / "demo_cases.json").is_file()
    report_template = (
        payload / "src" / "criterion_lens" / "report" / "templates" / "report.html.j2"
    )
    assert report_template.is_file()
    assert not (payload / ".env").exists()
    assert not (payload / "outputs").exists()

    code = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        "from app.ui import _examples; "
        "assert len(_examples()) == 3"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=payload,
        check=True,
        capture_output=True,
        text=True,
    )


def test_space_payload_refuses_existing_destination(tmp_path) -> None:
    destination = tmp_path / "payload"
    destination.mkdir()

    try:
        prepare_space_payload(destination, project_root=PROJECT_ROOT)
    except FileExistsError as error:
        assert "Refusing to overwrite" in str(error)
    else:
        raise AssertionError("Existing staging directory should not be overwritten")
