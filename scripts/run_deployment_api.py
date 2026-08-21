#!/usr/bin/env python3
"""Start one registered Module 14 service profile without device fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from governed_banking.deployment_config import DeploymentProfile
from governed_banking.observed_deployment_service import create_observed_deployment_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/deployment/native-mps.yaml"),
        help="Registered deployment profile (default: native macOS MPS)",
    )
    return parser.parse_args()


def main() -> None:
    profile = DeploymentProfile.from_yaml(parse_args().profile)
    app = create_observed_deployment_app(profile)
    uvicorn.run(
        app,
        host=profile.bind_host,
        port=profile.port,
        reload=False,
        workers=1,
        proxy_headers=False,
        forwarded_allow_ips="",
        access_log=False,
        log_level="warning",
        timeout_graceful_shutdown=int(profile.lifecycle.graceful_shutdown_seconds),
    )


if __name__ == "__main__":
    main()
