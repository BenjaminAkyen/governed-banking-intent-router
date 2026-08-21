#!/usr/bin/env python3
"""Start the authenticated Module 10 shadow API on the registered loopback address."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from governed_banking.api import ServiceConfig, build_runtime_app


def main() -> None:
    config_path = Path("configs/service.yaml")
    config = ServiceConfig.from_yaml(config_path)
    app = build_runtime_app(config_path)
    uvicorn.run(
        app,
        host=config.bind_host,
        port=config.default_port,
        reload=False,
        workers=1,
        proxy_headers=False,
        forwarded_allow_ips="",
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
