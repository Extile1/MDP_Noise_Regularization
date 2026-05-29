import os
import tempfile
import numpy as np
import gymnasium as gym
from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.wrappers import TimeLimit

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}

class NLinkReacherEnv(MujocoEnv, utils.EzPickle):
    """N-Link Reacher reproducing Reacher-v5 observation/reward logic."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        n_links=2,
        render_mode=None,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        reward_dist_weight=1.0,
        reward_control_weight=5.0,
        frame_skip=2,
    ):
        self.n_links = n_links
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        # Generate & save XML
        self.xml_content = self._generate_xml(n_links)
        self._temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        self._temp_file.write(self.xml_content)
        self._temp_file.close()

        # Observation: [cos(theta)(n), sin(theta)(n), target_pos(2), joint_vel(n), vector_to_target(3)]
        obs_dim = 2 * n_links + 2 + n_links + 3
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        utils.EzPickle.__init__(
            self,
            n_links,
            render_mode,
            default_camera_config,
            reward_dist_weight,
            reward_control_weight,
            frame_skip,
        )

        MujocoEnv.__init__(
            self,
            model_path=self._temp_file.name,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        reward, info = self._get_rew(action)
        terminated = False
        truncated = False
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _get_rew(self, action):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        dist_to_target = np.linalg.norm(vec)
        reward_dist = -dist_to_target * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight
        reward = reward_dist + reward_ctrl
        return reward, {"reward_dist": reward_dist, "reward_ctrl": reward_ctrl}

    def _get_obs(self):
        theta = self.data.qpos.flat[: self.n_links]
        target_pos = self.get_body_com("target")
        tip_pos = self.get_body_com("fingertip")
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                target_pos[:2],
                self.data.qvel.flat[: self.n_links],
                (tip_pos - target_pos),
            ]
        )

    def reset_model(self):
        # Randomize arm joints
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos

        # Randomize target in a disk
        total_reach = 0.2
        while True:
            target_x = self.np_random.uniform(low=-total_reach, high=total_reach)
            target_y = self.np_random.uniform(low=-total_reach, high=total_reach)
            if target_x**2 + target_y**2 < total_reach**2:
                break

        qpos[-2] = target_x
        qpos[-1] = target_y

        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _generate_xml(self, n):
        total_arm_length = 0.2
        link_len = total_arm_length / n

        xml = f"""
        <mujoco model="reacher_{n}links">
            <compiler angle="radian" inertiafromgeom="true"/>
            <default>
                <joint armature="1" damping="1" limited="true"/>
                <geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
            </default>
            <option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
            <worldbody>
                <geom conaffinity="0" contype="0" name="ground" pos="0 0 0" rgba="0.9 0.9 0.9 1" size="1 1 10" type="plane"/>
                <geom conaffinity="0" fromto="-.3 -.3 .01 .3 -.3 .01" name="sideS" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
                <geom conaffinity="0" fromto=" .3 -.3 .01 .3  .3 .01" name="sideE" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
                <geom conaffinity="0" fromto="-.3  .3 .01 .3  .3 .01" name="sideN" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
                <geom conaffinity="0" fromto="-.3 -.3 .01 -.3 .3 .01" name="sideW" rgba="0.9 0.4 0.6 1" size=".02" type="capsule"/>
                <geom conaffinity="0" contype="0" fromto="0 0 0 0 0 0.02" name="root" rgba="0.9 0.4 0.6 1" size=".011" type="cylinder"/>
        """

        indent = "\t" * 3
        xml += f'{indent}<body name="body0" pos="0 0 .01">\n'
        indent += "\t"

        for i in range(n):
            xml += f'{indent}<geom fromto="0 0 0 {link_len} 0 0" name="link{i}" rgba="0.0 0.4 0.6 1" size=".01" type="capsule"/>\n'
            if i == 0:
                xml += f'{indent}<joint axis="0 0 1" limited="false" name="joint{i}" pos="0 0 0" type="hinge"/>\n'
            else:
                xml += f'{indent}<joint axis="0 0 1" limited="true" name="joint{i}" pos="0 0 0" range="-3.0 3.0" type="hinge"/>\n'

            if i < n - 1:
                xml += f'{indent}<body name="body{i+1}" pos="{link_len} 0 0">\n'
                indent += "\t"
            else:
                xml += f'{indent}<body name="fingertip" pos="{link_len + 0.01} 0 0">\n'
                xml += f'{indent}\t<geom contype="0" name="fingertip" pos="0 0 0" rgba="0.0 0.8 0.6 1" size=".01" type="sphere"/>\n'
                xml += f'{indent}</body>\n'

        for _ in range(n):
            indent = indent[:-1]
            xml += f'{indent}</body>\n'

        xml += """
                <body name="target" pos=".1 -.1 .01">
                    <joint armature="0" axis="1 0 0" damping="0" limited="true" name="target_x" pos="0 0 0" range="-.27 .27" ref=".1" stiffness="0" type="slide"/>
                    <joint armature="0" axis="0 1 0" damping="0" limited="true" name="target_y" pos="0 0 0" range="-.27 .27" ref="-.1" stiffness="0" type="slide"/>
                    <geom conaffinity="0" contype="0" name="target" pos="0 0 0" rgba="0.9 0.2 0.2 1" size=".009" type="sphere"/>
                </body>
            </worldbody>
            <actuator>
        """

        for i in range(n):
            xml += f'\t\t<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="joint{i}"/>\n'

        xml += """
            </actuator>
        </mujoco>
        """
        return xml

    def close(self):
        super().close()
        if hasattr(self, "_temp_file") and self._temp_file is not None:
            if os.path.exists(self._temp_file.name):
                try:
                    os.remove(self._temp_file.name)
                except Exception:
                    pass

# =========================
# Noise Wrappers
# =========================

class StickyNoiseWrapper(gym.Wrapper):
    """With probability p, the agent remains in the current state."""
    def __init__(self, env, prob=0.1):
        super().__init__(env)
        self.prob = prob
        self.last_obs = None
        self.last_reward = 0.0 

    def reset(self, **kwargs):
        self.last_obs, info = self.env.reset(**kwargs)
        self.last_reward = 0.0
        return self.last_obs, info

    def step(self, action):
        if self.last_obs is not None and self.np_random.uniform() < self.prob:
            return self.last_obs, self.last_reward, False, False, {"noise": "sticky"}
        
        obs, reward, term, trunc, info = self.env.step(action)
        self.last_obs = obs
        self.last_reward = reward
        return obs, reward, term, trunc, info


class ResetNoiseWrapper(gym.Wrapper):
    """With probability p, the system crashes and resets."""
    def __init__(self, env, prob=0.01):
        super().__init__(env)
        self.prob = prob

    def step(self, action):
        if self.np_random.uniform() < self.prob:
            obs, info = self.env.reset()
            return obs, 0.0, False, False, {"noise": "reset"}
        return self.env.step(action)


class RandomActionWrapper(gym.Wrapper):
    """With probability p, a random action is executed."""
    def __init__(self, env, prob=0.1):
        super().__init__(env)
        self.prob = prob

    def step(self, action):
        if self.np_random.uniform() < self.prob:
            action = self.env.action_space.sample()
        return self.env.step(action)


# =========================
# Factory
# =========================

def make_reacher_env(n_links, episode_len, noise_type="none", noise_prob=0.0, render_mode=None, seed=None):
    """Factory to create env with optional noise."""
    env = NLinkReacherEnv(n_links=n_links, render_mode=render_mode)
    
    if noise_type != "none" and noise_prob > 0:
        if noise_type == "sticky":
            env = StickyNoiseWrapper(env, prob=noise_prob)
        elif noise_type == "reset":
            env = ResetNoiseWrapper(env, prob=noise_prob)
        elif noise_type == "random":
            env = RandomActionWrapper(env, prob=noise_prob)
            
    env = TimeLimit(env, max_episode_steps=episode_len)
    
    if seed is not None:
        env.reset(seed=seed)
    return env