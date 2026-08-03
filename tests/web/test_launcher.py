from __future__ import annotations

import argparse

import pytest

from rotarycam.web.launcher import DEFAULT_PORT, LOOPBACK_HOST, build_parser


def test_launcher_defaults_to_fixed_loopback_and_opens_browser() -> None:
    arguments = build_parser().parse_args([])

    assert LOOPBACK_HOST == "127.0.0.1"
    assert arguments.port == DEFAULT_PORT
    assert arguments.no_browser is False


def test_launcher_accepts_port_and_no_browser() -> None:
    arguments = build_parser().parse_args(["--port", "9000", "--no-browser"])

    assert arguments.port == 9000
    assert arguments.no_browser is True


@pytest.mark.parametrize("port", ["0", "65536"])
def test_launcher_rejects_unsafe_port(port: str) -> None:
    with pytest.raises(SystemExit) as caught:
        build_parser().parse_args(["--port", port])

    assert caught.value.code == 2


def test_launcher_port_validation_uses_argparse_error() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        from rotarycam.web.launcher import _port

        _port("70000")
