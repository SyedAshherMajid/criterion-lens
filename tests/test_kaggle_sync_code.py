import json
from types import SimpleNamespace

from scripts.kaggle import sync_code


def test_sync_code_embeds_freeze_identity_only_for_frozen_head(
    tmp_path, monkeypatch
) -> None:
    commit = "d" * 40

    def fake_run(arguments, **_kwargs):
        if "describe" in arguments:
            return SimpleNamespace(returncode=0, stdout="v1.0-frozen\n")
        return SimpleNamespace(returncode=0, stdout=f"{commit}\n")

    monkeypatch.setattr(sync_code.subprocess, "run", fake_run)
    sync_code._write_freeze_manifest(tmp_path)

    manifest = json.loads(
        (tmp_path / sync_code.FREEZE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest == {"tag": "v1.0-frozen", "commit": commit}


def test_sync_code_does_not_create_freeze_manifest_for_untagged_head(
    tmp_path, monkeypatch
) -> None:
    def fake_run(_arguments, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="main\n")

    monkeypatch.setattr(sync_code.subprocess, "run", fake_run)
    sync_code._write_freeze_manifest(tmp_path)

    assert not (tmp_path / sync_code.FREEZE_MANIFEST_NAME).exists()
