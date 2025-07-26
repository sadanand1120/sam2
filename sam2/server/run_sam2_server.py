#!/usr/bin/env python3

import argparse
import uvicorn

from sam2.server.api_servers import sam2_app, set_api_key


def main():
    parser = argparse.ArgumentParser(description="SAM2 Segmentation API Server")

    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8003,
                        help="Port to bind (default: 8003)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    parser.add_argument("--api-key", type=str, default="smfeats",
                        help="API key for authentication (required, default: smfeats)")
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Log level (default: info)")

    args = parser.parse_args()

    # Set API key (always required)
    set_api_key(args.api_key)
    print(f"🔐 Authentication enabled with API key: {args.api_key}")

    print(f"🚀 Starting SAM2 server on {args.host}:{args.port}")
    print(f"📖 Interactive docs: http://{args.host}:{args.port}/docs")

    if args.workers > 1:
        # For multiple workers, use import string
        uvicorn.run(
            "sam2.server.api_servers:sam2_app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=args.log_level
        )
    else:
        # For single worker, use app object directly
        uvicorn.run(
            sam2_app,
            host=args.host,
            port=args.port,
            log_level=args.log_level
        )


if __name__ == "__main__":
    main()
