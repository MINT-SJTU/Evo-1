import os
from typing import Optional, Sequence

from .config import parse_args


def main(category: str, argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(category, argv)

    # This must happen before importing LIBERO / MuJoCo.
    os.environ["MUJOCO_GL"] = config.mujoco_gl

    try:
        from .runtime import run_evaluation
    except ModuleNotFoundError as exc:
        if exc.name != "libero":
            raise
        raise SystemExit(
            "LIBERO is not importable outside its source directory. "
            "Apply libero-plus-eval.patch to LIBERO-Plus, then run "
            "'python -m pip install -e . --no-deps --force-reinstall'."
        ) from exc

    run_evaluation(config)
