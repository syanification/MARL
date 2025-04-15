import pyvirtualdisplay
from vmas.simulator.scenario import BaseScenario

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Line, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils, Y

from vmas import make_env

from IPython.display import Image

from vmas.simulator.scenario import BaseScenario
from typing import Union
import time
import torch
from vmas import make_env
from vmas.simulator.core import Agent

import copy
from typing import Callable, Optional
from benchmarl.environments import VmasTask
from benchmarl.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, VmasEnv

display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
display.start()

class CustomVmasEnv(VmasEnv):
    def __init__(self, *args, scenario: BaseScenario, **kwargs):
        super().__init__(*args, scenario=scenario, **kwargs)
        self.scenario = scenario  # Store the scenario for custom logic

    def step(self, actions):
        # Call scenario.step() before the environment step
        self.scenario.step()
        # print("step")
        # Proceed with the normal environment step
        return super().step(actions)

    def pre_step(self, actions):
        # This gets called before the actions are applied
        # print("pre_step")
        if self.frozen_agent is not None:
            for env_idx in range(self.world.batch_dim):
                actions[self.frozen_agent][env_idx] = torch.zeros_like(actions[self.frozen_agent][env_idx])
        return actions

class MyScenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        print("make_world called for scenario")
        self.n_agents = kwargs.pop("n_agents", 3)
        self.package_mass = kwargs.pop("package_mass", 5)
        self.random_package_pos_on_line = kwargs.pop("random_package_pos_on_line", True)
        self.k = kwargs.pop("k", 5)  # Add k as a parameter with a default value of 5
        ScenarioUtils.check_kwargs_consumed(kwargs)

        assert self.n_agents > 1

        self.line_length = 0.8
        self.agent_radius = 0.03

        self.shaping_factor = 100
        self.fall_reward = -10

        self.visualize_semidims = False

        # Initialize frozen agent tracking
        self.frozen_agent = None
        self.freeze_counter = 0
        self.original_color = None  # To store the original color of the frozen agent

        # Make world
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.7,
                color=Color.RED,  # Default color for agents
            )
            world.add_agent(agent)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # Add landmarks

        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        self.floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        return world

    # def step(self):
    #     print("scenario.step() called")  # Debug print
    #     # Randomly select an agent to freeze if no agent is currently frozen
    #     if self.frozen_agent is None and torch.randint(0, 3, (1,)).item() == 0:  # Random interval
    #         self.frozen_agent = torch.randint(0, self.n_agents, (1,)).item()
    #         self.freeze_counter = self.k

    #         # Change the color of the frozen agent to cyan
    #         frozen_agent = self.world.agents[self.frozen_agent]
    #         self.original_color = frozen_agent.color  # Save the original color
    #         frozen_agent.color = Color.BLUE

    #     # Decrease freeze counter and unfreeze the agent if the counter reaches 0
    #     if self.frozen_agent is not None:
    #         print(f"Agent {self.frozen_agent} frozen for {self.freeze_counter} steps")
    #         self.freeze_counter -= 1
    #         if self.freeze_counter <= 0:
    #             # Revert the color of the agent back to its original color
    #             frozen_agent = self.world.agents[self.frozen_agent]
    #             frozen_agent.color = self.original_color
    #             self.frozen_agent = None

    # def override_action(self, agent: Agent, action: torch.Tensor) -> torch.Tensor:
    #     # Override the action of the frozen agent with a "zero-acceleration" action
    #     if self.frozen_agent is not None:
    #         print(f"override_action called for agent {agent.name}, frozen agent: {self.frozen_agent}")
    #         if agent == self.world.agents[self.frozen_agent]:
    #             return torch.zeros_like(action)  # Zero-acceleration action
    #     return action

    def reset_world_at(self, env_index: int = None):
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    (
                        -self.line_length / 2 + self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                    (
                        self.line_length / 2 - self.package.shape.radius
                        if self.random_package_pos_on_line
                        else 0.0
                    ),
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -self.agent_radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )

        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - self.agent_radius,
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.compute_on_the_ground()
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def observation(self, agent: Agent):
        # Get positions of all entities in this agent's reference frame
        base_observation = torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                agent.state.pos - self.line.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.package.state.vel,
                self.line.state.vel,
                self.line.state.ang_vel,
                self.line.state.rot % torch.pi,
            ],
            dim=-1,
        )

        # Collect positions and velocities of all agents
        all_positions = torch.stack([a.state.pos for a in self.world.agents], dim=0)
        all_velocities = torch.stack([a.state.vel for a in self.world.agents], dim=0)

        # Compute mean and variance for positions and velocities
        mean_position = torch.mean(all_positions, dim=0)
        variance_position = torch.var(all_positions, dim=0)
        mean_velocity = torch.mean(all_velocities, dim=0)
        variance_velocity = torch.var(all_velocities, dim=0)

        # Concatenate the mean and variance to the observation
        extended_observation = torch.cat(
            [base_observation, mean_position, variance_position, mean_velocity, variance_velocity],
            dim=-1,
        )

        return extended_observation

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            # Call step logic at the beginning of reward calculation for the first agent
            if self.frozen_agent is None and torch.randint(0, 10, (1,)).item() == 0:
                self.frozen_agent = torch.randint(0, self.n_agents, (1,)).item()
                self.freeze_counter = self.k
                frozen_agent = self.world.agents[self.frozen_agent]
                self.original_color = frozen_agent.color
                frozen_agent.color = Color.BLUE
                # print(f"Agent {self.frozen_agent} frozen for {self.k} stepss")

            if self.frozen_agent is not None:
                self.freeze_counter -= 1
                if self.freeze_counter <= 0:
                    frozen_agent = self.world.agents[self.frozen_agent]
                    frozen_agent.color = self.original_color
                    self.frozen_agent = None
                    # print(f"Agent unfrozen")

            # Original reward calculation code
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0
            # Rest of the reward logic...

            self.compute_on_the_ground()
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

            self.ground_rew[self.on_the_ground] = self.fall_reward

            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

        return self.ground_rew + self.pos_rew
    
    def done(self):
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )
    
    def info(self, agent: Agent):
        info = {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}
        return info
    
    def compute_on_the_ground(self):
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)
    
    def process_action(self, agent: Agent):
        # If this agent is frozen, zero out its acceleration
        if self.frozen_agent is not None and agent == self.world.agents[self.frozen_agent]:
            agent.action.u = torch.zeros_like(agent.action.u)
        return

env = make_env(
    scenario = MyScenario(),
    num_envs = 8,
    device = "cuda",
    seed = 0,

    #Optional:
    continuous_actions = False,
    max_steps = 100,

    #Env Specific
    n_agents = 4,
    package_mass = 5
)

actions = env.get_random_actions()
# print(actions)

obs, rews, dones, info = env.step(actions)

# print(f"Obs length: {len(obs)}, observation of agent 0:\n{obs[0]}")
# print(f"Rewards length: {len(rews)}, reward of agent 0:\n{rews[0]}")
# print(dones)


def use_vmas_env(
    render: bool,
    num_envs: int,
    n_steps: int,
    device: str,
    scenario: Union[str, BaseScenario],
    continuous_actions: bool,
    **kwargs
):
    """Example function to use a vmas environme
    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.
nt.

    This is a simplification of the function in `vmas.examples.use_vmas_env.py`.

    Args:
        continuous_actions (bool): Whether the agents have continuous or discrete actions
        scenario (str, BaseScenario): Name of scenario or scenario class
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        num_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done

    """

    scenario_name = scenario if isinstance(scenario, str) else scenario.__class__.__name__

    env = make_env(
        scenario=scenario,
        num_envs=num_envs,
        device=device,
        continuous_actions=continuous_actions,
        seed=0,
        # Environment specific variables
        **kwargs
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0

    for s in range(n_steps):
        step += 1
        print(f"Step {step}")
        scenario.step()

        actions = []
        for i, agent in enumerate(env.agents):
            # Get the agent's action
            action = env.get_random_action(agent)

            # Override the action if the agent is frozen
            action = scenario.override_action(agent, action)

            actions.append(action)

        # Step the environment
        obs, rews, dones, info = env.step(actions)

        # Render if required
        if render:
            frame = env.render(mode="rgb_array")
            frame_list.append(frame)

    total_time = time.time() - init_time
    print(
        f"It took: {total_time}s for {n_steps} steps of {num_envs} parallel environments on device {device} "
        f"for {scenario_name} scenario."
    )

    if render:
        from moviepy import ImageSequenceClip
        fps=30
        clip = ImageSequenceClip(frame_list, fps=fps)
        clip.write_gif(f'{scenario_name}.gif', fps=fps)

'''
use_vmas_env(
    render=True,
    num_envs=8,
    n_steps=250,
    device="cuda",
    scenario=MyScenario(),
    continuous_actions=True,
    # Scenario kwargs
    n_agents=4,
    k=50
)

Image(f'{MyScenario.__name__}.gif')
'''

def get_env_fun(
    self,
    num_envs: int,
    continuous_actions: bool,
    seed: Optional[int],
    device: DEVICE_TYPING,
) -> Callable[[], EnvBase]:
    print("get_env_fun called with:", self, num_envs, continuous_actions, seed, device)
    config = copy.deepcopy(self.config)
    if self is VmasTask.BALANCE:
        scenario = MyScenario()  # Use the custom scenario
        print("SWAPPED OUT SCENARIO")
    else:
        scenario = self.name.lower()
    return lambda: CustomVmasEnv(  # Use the custom VmasEnv class
        scenario=scenario,
        num_envs=num_envs,
        continuous_actions=continuous_actions,
        seed=seed,
        device=device,
        categorical_actions=True,
        clamp_actions=True,
        **config,
    )

try:
    from benchmarl.environments import VmasClass
    VmasClass.get_env_fun = get_env_fun
except ImportError:
    VmasTask.get_env_fun = get_env_fun

train_device = "cuda" # @param {"type":"string"}
vmas_device = "cuda" # @param {"type":"string"}

#EXPERIMENT CONFIG
from benchmarl.experiment import ExperimentConfig

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml() # We start by loading the defaults

# Override devices
experiment_config.sampling_device = vmas_device
experiment_config.train_device = train_device

experiment_config.max_n_frames = 10_000_000 # Number of frames before training ends
experiment_config.gamma = 0.99
experiment_config.on_policy_collected_frames_per_batch = 60_000 # Number of frames collected each iteration
experiment_config.on_policy_n_envs_per_worker = 600 # Number of vmas vectorized enviornemnts (each will collect 100 steps, see max_steps in task_config -> 600 * 100 = 60_000 the number above)
experiment_config.on_policy_n_minibatch_iters = 45
experiment_config.on_policy_minibatch_size = 4096
experiment_config.evaluation = True
experiment_config.render = True
experiment_config.share_policy_params = True # Policy parameter sharing on
experiment_config.evaluation_interval = 120_000 # Interval in terms of frames, will evaluate every 120_000 / 60_000 = 2 iterations
experiment_config.evaluation_episodes = 200 # Number of vmas vectorized enviornemnts used in evaluation
experiment_config.loggers = ["wandb"] # Log to csv, usually you should use wandb

# Loads from "benchmarl/conf/task/vmas/balance.yaml"
task = VmasTask.BALANCE.get_from_yaml()


print(task.config)

# We override the balance config with ours
task.config = {
    "n_agents": 4,
    "random_package_pos_on_line": True,
    "package_mass": 5,
    "k": 35, # Number of steps the agent is frozen
    "max_steps": 1000, # Number of steps each agent can take
}

# print(task.config)

from benchmarl.algorithms import IppoConfig

# We can load from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config = IppoConfig.get_from_yaml()

# Or create it from scratch
algorithm_config = IppoConfig(
        share_param_critic=True, # Critic param sharing on
        clip_epsilon=0.2,
        entropy_coef=0.001, # We modify this, default is 0
        critic_coef=1,
        loss_critic_type="l2",
        lmbda=0.9,
        scale_mapping="biased_softplus_1.0", # Mapping for standard deviation
        use_tanh_normal=True,
        minibatch_advantage=False,
    )

from benchmarl.models.mlp import MlpConfig

model_config = MlpConfig(
        num_cells=[256, 256], # Two layers with 256 neurons each
        layer_class=torch.nn.Linear,
        activation_class=torch.nn.Tanh,
    )

# Loads from "benchmarl/conf/model/layers/mlp.yaml" (in this case we use the defaults so it is the same)
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

from benchmarl.experiment import Experiment

experiment_config.max_n_frames = 50_000_000 # Runs one iteration, change to 50_000_000 for full training
# experiment_config.on_policy_n_envs_per_worker = 60 # Remove this line for full training
# experiment_config.on_policy_n_minibatch_iters = 1 # Remove this line for full training

experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)
experiment.run()

from torchrl.envs import VmasEnv



