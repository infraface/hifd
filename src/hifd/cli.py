"""Command-line interface for HiFD. Installed as console_script ``hifd``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

from . import __version__
from . import io as hio
from . import pipeline as pl
from .composite import load_profiles_yaml
from .constants import DEFAULT_CONSTANTS, DEFAULT_PROFILES


def _merged_constants(path: str | None) -> dict[str, float]:
    base = dict(DEFAULT_CONSTANTS)
    if path is not None:
        base.update(hio.load_constants_yaml(path))
    return base


def _profiles(path: str | None) -> Mapping[str, Mapping[str, float]]:
    if path is None:
        return DEFAULT_PROFILES
    return load_profiles_yaml(path)


# ----- Commands ---------------------------------------------------------------

def cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    methods = hio.load_methods_yaml(args.methods)
    estimators = hio.load_estimators_yaml(args.estimators)
    data = Path(args.data_dir)
    orig_base = data / methods["original"]["base_path"]

    errors: list[str] = []
    for m in methods["methods"]:
        method_base = data / m["base_path"]
        if not method_base.exists():
            errors.append(f"{m['name']}: method dir not found: {method_base}")
            continue
        for task_name, task_cfg in estimators["tasks"].items():
            for est in task_cfg["estimators"]:
                op = orig_base / est["orig_json"] if "orig_json" in est else None
                dp = method_base / est["deid_json"]
                if op is not None and not op.exists():
                    errors.append(f"{m['name']}/{task_name}: orig not found: {op}")
                    continue
                if not dp.exists():
                    errors.append(f"{m['name']}/{task_name}: deid not found: {dp}")
                    continue
                if op is not None:
                    try:
                        hio.validate_pair(str(op), str(dp))
                    except ValueError as e:
                        errors.append(f"{m['name']}/{task_name}/{est['name']}: {e}")

    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_compute_agreements(args: argparse.Namespace) -> int:
    pl.compute_agreements(
        methods_config=args.methods,
        data_dir=args.data_dir,
        out_dir=args.out,
        estimators_config=args.estimators,
        constants=_merged_constants(args.constants),
    )
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    pl.aggregate(args.input, args.out)
    return 0


def cmd_compose(args: argparse.Namespace) -> int:
    pl.compose(
        in_dir=args.input,
        out_path=args.out,
        profiles=_profiles(args.profiles),
        constants=_merged_constants(args.constants),
    )
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    pl.run_pipeline(
        methods_config=args.methods,
        data_dir=args.data_dir,
        out_dir=args.out,
        estimators_config=args.estimators,
        constants=_merged_constants(args.constants),
        profiles=_profiles(args.profiles),
    )
    return 0


# ----- Parser -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hifd", description="HiFD metric CLI")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("validate")
    sp.add_argument("--methods", required=True)
    sp.add_argument("--estimators", required=True)
    sp.add_argument("--data-dir", default="data")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("compute-agreements")
    sp.add_argument("--methods", required=True)
    sp.add_argument("--estimators", required=True)
    sp.add_argument("--constants", default=None)
    sp.add_argument("--data-dir", default="data")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_compute_agreements)

    sp = sub.add_parser("aggregate")
    sp.add_argument("--in", dest="input", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_aggregate)

    sp = sub.add_parser("compose")
    sp.add_argument("--in", dest="input", required=True)
    sp.add_argument("--profiles", default=None)
    sp.add_argument("--constants", default=None)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_compose)

    sp = sub.add_parser("run-all")
    sp.add_argument("--methods", required=True)
    sp.add_argument("--estimators", default=None)
    sp.add_argument("--profiles", default=None)
    sp.add_argument("--constants", default=None)
    sp.add_argument("--data-dir", default="data")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_run_all)

    sp = sub.add_parser("version")
    sp.set_defaults(func=cmd_version)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as e:
        print(f"IO error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
