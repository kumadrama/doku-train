"""Local prepared-input verification entry point."""

import argparse
import json
from pathlib import Path

from .comm.training_pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--local-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-checkpoint")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.evaluate_checkpoint:
        from .comm.training_pipeline import evaluate_checkpoint

        if not args.prepared or args.resume:
            parser.error("Evaluation requires --prepared and does not accept --resume")
        evaluate_checkpoint(
            config, args.prepared, args.evaluate_checkpoint, args.output, args.split
        )
    elif args.prepared:
        run(config, args.prepared, args.output, args.resume)
    else:
        from .experiment import execute

        if args.resume:
            parser.error("Manifest entry requires frozen prepared input for resume; use --prepared")
        execute(config, args.output, local_root=Path(args.local_root) if args.local_root else None)


if __name__ == "__main__":
    main()
