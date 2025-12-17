"""The Loom - Main entry point.

A research-focused model server exposing hidden states for AI interpretability.
Part of the Weaver ecosystem for multi-agent orchestration and conveyance
measurement.

Usage:
    loom                        # Start with defaults (HTTP)
    loom --port 8080            # Custom HTTP port
    loom --transport unix       # Unix socket only
    loom --transport both       # HTTP and Unix socket
    loom --config my.yaml       # Custom config
    loom --preload model-id     # Preload a model
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import uvicorn

from .config import Config, load_config, set_config
from .transport.http import create_http_app


def setup_logging(config: Config) -> None:
    """
    Configure the root logger and attach handlers according to the given configuration.
    
    Creates a console StreamHandler writing to stdout with the configured format. If a file path is specified, creates any missing parent directories and adds a FileHandler using the same format. Sets the root logger level from config.logging.level and attaches all created handlers. Lowers log levels for common third-party libraries (httpx, httpcore, urllib3, transformers) to reduce noise.
    
    Parameters:
        config (Config): Configuration containing logging settings (e.g., `logging.format`, `logging.level`, and optional `logging.file`).
    """
    log_config = config.logging

    # Set up handlers
    handlers: list[logging.Handler] = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_config.format))
    handlers.append(console_handler)

    # File handler if configured
    if log_config.file:
        log_path = Path(log_config.file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_config.format))
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_config.level.upper()))
    for handler in handlers:
        root_logger.addHandler(handler)

    # Reduce noise from third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Loom server.
    
    Parses options for configuration file path, transport mode (http, unix, both), HTTP host and port, Unix socket path, models to preload, default CUDA device, logging level, and the auto-reload flag. The parser also provides example usage and documents environment-variable overrides in its epilog.
    
    Returns:
        argparse.Namespace: Parsed CLI options with attributes such as `config`, `transport`, `host`, `port`, `unix_socket`, `preload`, `device`, `log_level`, and `reload`.
    """
    parser = argparse.ArgumentParser(
        description="The Loom - Hidden state extraction for AI research (part of the Weaver ecosystem)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    loom                              Start with default config (HTTP)
    loom --port 9000                  Custom HTTP port
    loom --transport unix             Unix socket only
    loom --transport both             HTTP and Unix socket
    loom --unix-socket /tmp/loom.sock Custom socket path
    loom --config config.yaml         Custom config file
    loom --preload llama3.1:8b        Preload a model at startup

Environment Variables:
    LOOM_SERVER__HTTP_PORT            Override HTTP port
    LOOM_SERVER__TRANSPORT            Override transport (http, unix, both)
    LOOM_SERVER__UNIX_SOCKET          Override Unix socket path
    LOOM_GPU__DEFAULT_DEVICE          Override default GPU device
    LOOM_LOGGING__LEVEL               Override log level
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to configuration file (YAML)",
    )

    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["http", "unix", "both"],
        default=None,
        help="Transport type: http, unix, or both (default: http)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="HTTP server host (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help="HTTP server port (default: 8080)",
    )

    parser.add_argument(
        "--unix-socket",
        "-u",
        type=str,
        default=None,
        help="Unix socket path (default: /tmp/loom.sock)",
    )

    parser.add_argument(
        "--preload",
        type=str,
        nargs="*",
        default=None,
        help="Model(s) to preload at startup",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Default CUDA device (e.g., cuda:0)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    return parser.parse_args()


async def run_http_server(
    app: Any,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """
    Start an HTTP server using Uvicorn to serve the given ASGI app on the specified host and port.
    
    Parameters:
        app (Any): An ASGI application instance (for example, a FastAPI or Starlette app).
        host (str): Network interface to bind the HTTP server to.
        port (int): TCP port to listen on.
        log_level (str): Logging level passed to Uvicorn (e.g., "info", "warning", "debug").
    """
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_unix_server(
    app: Any,
    socket_path: str,
    log_level: str,
) -> None:
    """
    Start a Uvicorn server bound to the given Unix domain socket.
    
    Removes an existing socket file at `socket_path` before binding and runs the server until it stops.
    
    Parameters:
        app: ASGI application to serve.
        socket_path (str): Filesystem path for the Unix domain socket to bind.
        log_level (str): Uvicorn log level (e.g., "info", "warning", "error").
    """
    # Remove existing socket file if present
    socket_file = Path(socket_path)
    if socket_file.exists():
        socket_file.unlink()

    config = uvicorn.Config(
        app,
        uds=socket_path,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_both_servers(
    app: Any,
    host: str,
    port: int,
    socket_path: str,
    log_level: str,
) -> None:
    """
    Start HTTP and Unix socket servers concurrently and wait for both to finish.
    
    This function launches the HTTP server on the specified host and port and the Unix socket server at the specified socket_path, running them concurrently until both complete.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting HTTP server on {host}:{port}")
    logger.info(f"Starting Unix socket server on {socket_path}")

    await asyncio.gather(
        run_http_server(app, host, port, log_level),
        run_unix_server(app, socket_path, log_level),
    )


def preload_models(app: Any, model_ids: list[str]) -> None:
    """
    Preload the specified models into the application's model manager.
    
    Preloads each model identified in `model_ids` by calling the app's model manager loader. Errors encountered while loading an individual model are caught and logged; they do not stop the remaining models from being processed.
    
    Parameters:
        app (Any): Application object with a `state.model_manager` providing `get_or_load(model_id)`.
        model_ids (list[str]): Sequence of model identifier strings to preload.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preloading {len(model_ids)} model(s)...")

    # Access model_manager from app state
    model_manager = app.state.model_manager

    for model_id in model_ids:
        try:
            logger.info(f"Preloading: {model_id}")
            model_manager.get_or_load(model_id)
            logger.info(f"Preloaded: {model_id}")
        except Exception as e:
            logger.error(f"Failed to preload {model_id}: {e}")


def main() -> None:
    """
    Start the Loom server using command-line arguments and configuration.
    
    Loads configuration (optionally overridden by CLI), applies it as the global config, configures logging, constructs the HTTP/FastAPI app, optionally preloads models, and launches the server using the configured transport mode ('http', 'unix', or 'both'). Exits with status 1 if the transport value is unrecognized.
    """
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    if args.transport is not None:
        config.server.transport = args.transport
    if args.host is not None:
        config.server.http_host = args.host
    if args.port is not None:
        config.server.http_port = args.port
    if args.unix_socket is not None:
        config.server.unix_socket = args.unix_socket
    if args.device is not None:
        device_idx = int(args.device.split(":")[-1]) if ":" in args.device else int(args.device)
        config.gpu.default_device = device_idx
    if args.log_level is not None:
        config.logging.level = args.log_level
    if args.preload is not None:
        config.models.preload = args.preload

    # Set global config
    set_config(config)

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Print startup banner
    logger.info("=" * 60)
    logger.info("The Loom - Hidden State Server")
    logger.info("Part of the Weaver Ecosystem")
    logger.info("=" * 60)
    logger.info(f"Transport: {config.server.transport}")
    if config.server.transport in ("http", "both"):
        logger.info(f"HTTP: {config.server.http_host}:{config.server.http_port}")
    if config.server.transport in ("unix", "both"):
        logger.info(f"Unix Socket: {config.server.unix_socket}")
    logger.info(f"Default device: cuda:{config.gpu.default_device}")
    logger.info(f"Max loaded models: {config.models.max_loaded}")
    logger.info("=" * 60)

    # Create the app
    app = create_http_app(config)

    # Preload models if specified
    if config.models.preload:
        preload_models(app, config.models.preload)

    # Get log level for uvicorn
    log_level = config.logging.level.lower()

    # Run the server based on transport type
    transport = config.server.transport

    if transport == "http":
        # HTTP only - use simple uvicorn.run
        uvicorn.run(
            app,
            host=config.server.http_host,
            port=config.server.http_port,
            log_level=log_level,
            reload=args.reload,
        )

    elif transport == "unix":
        # Unix socket only
        socket_path = config.server.unix_socket

        # Remove existing socket file if present
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        uvicorn.run(
            app,
            uds=socket_path,
            log_level=log_level,
            reload=args.reload,
        )

    elif transport == "both":
        # Both HTTP and Unix socket - run concurrently
        if args.reload:
            logger.warning("--reload is not supported with 'both' transport, ignoring")

        asyncio.run(
            run_both_servers(
                app,
                config.server.http_host,
                config.server.http_port,
                config.server.unix_socket,
                log_level,
            )
        )

    else:
        logger.error(f"Unknown transport: {transport}")
        sys.exit(1)


if __name__ == "__main__":
    main()