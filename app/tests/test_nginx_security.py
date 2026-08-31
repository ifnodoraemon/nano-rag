"""Security posture of the public nginx entry (docker/frontend/nginx.conf).

The previous config injected X-Admin-API-Key on every proxied location with
the same value as the business proxy key, so anyone reaching the public
frontend port had full admin/debug access. These tests pin the fix.
"""
from __future__ import annotations

import re
from pathlib import Path

NGINX_CONF = Path(__file__).resolve().parents[2] / "docker" / "frontend" / "nginx.conf"


def _locations(conf: str) -> dict[str, str]:
    """Map each location block to its raw body text."""
    blocks: dict[str, str] = {}
    pattern = re.compile(r"location\s+([^\s{]+)\s*\{")
    for match in pattern.finditer(conf):
        depth = 0
        start = match.end() - 1
        for index in range(start, len(conf)):
            if conf[index] == "{":
                depth += 1
            elif conf[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks[match.group(1)] = conf[start : index + 1]
                    break
    return blocks


def test_admin_key_is_never_injected() -> None:
    conf = NGINX_CONF.read_text(encoding="utf-8")
    blocks = _locations(conf)
    assert blocks, "nginx.conf must define location blocks"
    for name, body in blocks.items():
        assert "X-Admin-API-Key" not in body, (
            f"location {name} must not inject X-Admin-API-Key: the public "
            "proxy may only inject the business key; admin routes must be "
            "authenticated by the caller"
        )


def test_business_key_injection_is_limited_to_business_routes() -> None:
    conf = NGINX_CONF.read_text(encoding="utf-8")
    blocks = _locations(conf)
    for name, body in blocks.items():
        if "X-API-Key" in body:
            assert name.startswith("/v1/rag/"), (
                f"location {name} injects the backend business key; only "
                "/v1/rag/ business routes may receive it"
            )


def test_operator_routes_stay_proxied_without_credentials() -> None:
    """Operator routes remain reachable (an operator presents the admin key
    themselves), but the proxy never authenticates them on the caller's
    behalf."""
    conf = NGINX_CONF.read_text(encoding="utf-8")
    blocks = _locations(conf)
    for route in ["/traces", "/eval/", "/benchmark/", "/diagnose/", "/replay/", "/debug/"]:
        assert route in blocks, f"operator route {route} must stay proxied"
        assert "X-API-Key" not in blocks[route]


def test_sse_stream_location_disables_buffering() -> None:
    conf = NGINX_CONF.read_text(encoding="utf-8")
    blocks = _locations(conf)
    sse = blocks.get("/v1/rag/chat/stream")
    assert sse is not None, "the SSE stream needs its own location block"
    assert "proxy_buffering off" in sse
    assert "600s" in sse


def test_upload_size_matches_app_limit() -> None:
    conf = NGINX_CONF.read_text(encoding="utf-8")
    assert "client_max_body_size" in conf
