import sys
from unittest.mock import Mock

import pytest

from app import __main__ as app_main


@pytest.mark.parametrize(("arguments", "share"), [([], False), (["--share"], True)])
def test_main_launches_local_app_and_opens_browser(
    monkeypatch,
    arguments: list[str],
    share: bool,
) -> None:
    launch = Mock()
    monkeypatch.setattr(app_main, "build_app", lambda: Mock(launch=launch))
    monkeypatch.setattr(sys, "argv", ["python -m app", *arguments])

    app_main.main()

    launch.assert_called_once_with(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=share,
    )
