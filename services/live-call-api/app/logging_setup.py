"""Central structured logging.

Every log line is one JSON object on stdout — that's the "central logging
infrastructure" for this stage of the project: docker compose already
aggregates every container's stdout, and Dozzle (see docker-compose.yml)
gives a zero-config web UI over it, filterable by field since every line
is real JSON. See docs/testing.md, "Reading the logs".

Swapping this for a real log-shipping stack (Loki+Grafana, ELK) later is a
matter of pointing the docker log driver or a sidecar at stdout — nothing
in application code needs to change, because application code only ever
calls get_logger() and binds fields onto it.
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(**initial_context: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(**initial_context)
