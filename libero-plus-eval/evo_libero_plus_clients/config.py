import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


SUITE_MAX_STEPS = {
    "libero_spatial": 660,
    "libero_object": 840,
    "libero_goal": 900,
    "libero_10": 1560,
}

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "logs" / "evo1_libero_plus"
DEFAULT_CKPT_TEMPLATE = (
    "step_80000_official/"
    "{category}_total_h{horizon}_S{seed}_test_true"
)


@dataclass(frozen=True)
class ClientConfig:
    category: str
    horizon: int
    server_url: str
    task_suites: Tuple[str, ...]
    max_steps_override: Optional[int]
    output_dir: Path
    ckpt_template: str
    num_episodes: int
    seed: int
    mujoco_gl: str

    @property
    def ckpt_name(self) -> str:
        return self.ckpt_template.format(
            category=self.category,
            horizon=self.horizon,
            seed=self.seed,
        )

    @property
    def suite_label(self) -> str:
        if len(self.task_suites) == 1:
            return self.task_suites[0].removeprefix("libero_")
        return "multi_suite"

    @property
    def log_file(self) -> Path:
        return self.output_dir / self.suite_label / f"{self.ckpt_name}.txt"

    @property
    def video_log_dir(self) -> Path:
        return self.output_dir / self.suite_label / self.ckpt_name

    def max_steps_for(self, suite: str) -> int:
        if self.max_steps_override is not None:
            return self.max_steps_override
        return SUITE_MAX_STEPS[suite]


def parse_args(category: str, argv: Optional[Sequence[str]] = None) -> ClientConfig:
    parser = argparse.ArgumentParser(
        description=f"Run the LIBERO-Plus {category} perturbation client."
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=tuple(SUITE_MAX_STEPS),
        default=["libero_spatial"],
        help="One or more LIBERO task suites (default: libero_spatial).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the built-in maximum steps for every selected suite.",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("LIBERO_PLUS_SERVER_URL", "ws://127.0.0.1:9003"),
        help="Policy WebSocket URL (env: LIBERO_PLUS_SERVER_URL).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("LIBERO_PLUS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        help="Root directory for logs and videos (env: LIBERO_PLUS_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--ckpt-template",
        default=os.environ.get("LIBERO_PLUS_CKPT_TEMPLATE", DEFAULT_CKPT_TEMPLATE),
        help="Output name template; supports {category}, {horizon}, and {seed}.",
    )
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mujoco-gl",
        choices=("egl", "osmesa", "glfw"),
        default=os.environ.get("MUJOCO_GL", "egl"),
        help="MuJoCo rendering backend (default: MUJOCO_GL or egl).",
    )
    namespace = parser.parse_args(argv)

    if namespace.horizon <= 0:
        parser.error("--horizon must be greater than zero")
    if namespace.num_episodes <= 0:
        parser.error("--num-episodes must be greater than zero")
    if namespace.max_steps is not None and namespace.max_steps <= 0:
        parser.error("--max-steps must be greater than zero")

    try:
        namespace.ckpt_template.format(
            category=category,
            horizon=namespace.horizon,
            seed=namespace.seed,
        )
    except (KeyError, ValueError) as exc:
        parser.error(f"invalid --ckpt-template: {exc}")

    return ClientConfig(
        category=category,
        horizon=namespace.horizon,
        server_url=namespace.server_url,
        task_suites=tuple(namespace.suite),
        max_steps_override=namespace.max_steps,
        output_dir=namespace.output_dir.expanduser().resolve(),
        ckpt_template=namespace.ckpt_template,
        num_episodes=namespace.num_episodes,
        seed=namespace.seed,
        mujoco_gl=namespace.mujoco_gl,
    )
