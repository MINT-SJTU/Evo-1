from typing import List, Optional 

import aloha.image_tools as image_tools


import aloha.real_env as _real_env


class AlohaRealEnvironment():
    """An environment for an Aloha robot on real hardware."""

    def __init__(
        self,
        reset_position: Optional[List[float]] = None, 
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        self._env = _real_env.make_real_env(init_node=True, reset_position=reset_position)
        self._render_height = render_height
        self._render_width = render_width

        self._ts = None

    # @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    # @override
    def is_episode_complete(self) -> bool:
        return False

    # @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation
        for k in list(obs["images"].keys()):
            if "_depth" in k:
                del obs["images"][k]

        for cam_name in obs["images"]:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["images"][cam_name], self._render_height, self._render_width)
            )
            # obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
            obs["images"][cam_name] = img

        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    # @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
