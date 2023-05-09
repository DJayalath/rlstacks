import argparse

import jax
import jumanji
from jumanji.wrappers import JumanjiToGymWrapper
import random

import matplotlib
import wandb
import ray
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune import register_env
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

from policy_net import PolicyNetwork
from stack_net import StackNetwork

matplotlib.use("QtAgg")

# for i in range(100):
#     # (Optional) Render the env state
#     env.render(state)
#
#     # Interact with the (jit-able) environment
#     action = env.action_spec().generate_value()          # Action selection (dummy value here)
#     action = random.randint(0, 3)
#     print(action)
#     state, timestep = jax.jit(env.step)(state, action)   # Take a step and observe the next state and time step

def train(
    scenario_name,
    encoder,
    encoding_dim,
    encoder_file,
    encoder_loss,
    use_stack,
    use_proj,
    no_stand,
    train_batch_size,
    sgd_minibatch_size,
    # max_steps,
    training_iterations,
    num_workers,
    num_envs,
    num_cpus_per_worker,
    seed,
    render_env,
    wandb_name,
    device,
    share_parameters,
    vmas_device="cpu",
):
    num_envs_per_worker = num_envs
    rollout_fragment_length = 100

    # Instantiate a Jumanji environment using the registry
    env = jumanji.make('Maze-v0')
    env = JumanjiToGymWrapper(env, seed=0)
    env.reset()

    # FIXME: Had to hack the wrapper to set info = {} to be compatible with Rllib

    register_env('Maze-v0', lambda config: env)

    if use_stack is True:
        ModelCatalog.register_custom_model("policy_net", StackNetwork)
    else:
        ModelCatalog.register_custom_model("policy_net", PolicyNetwork)

    # Train policy!
    ray.tune.run(
        PPOTrainer,
        stop={"training_iteration": training_iterations},
        checkpoint_freq=1,
        keep_checkpoints_num=2,
        checkpoint_at_end=True,
        checkpoint_score_attr="episode_reward_mean",
        callbacks=[
            WandbLoggerCallback(
                project=f"rl_stacks",
                name=f"{wandb_name}-seed-{seed}",
                entity="dhjayalath",
                api_key="",
            )
        ],
        config={
            "seed": seed,
            "framework": "torch",
            "env": 'Maze-v0',
            "render_env": render_env,
            # "kl_coeff": 0.01,
            # "kl_target": 0.01,
            # "lambda": 0.9,
            # "clip_param": 0.2,
            # "vf_loss_coeff": 1,
            # "vf_clip_param": float("inf"),
            # "entropy_coeff": 0.01,
            "train_batch_size": train_batch_size,
            # Should remain close to max steps to avoid bias
            "rollout_fragment_length": rollout_fragment_length,
            "sgd_minibatch_size": sgd_minibatch_size,
            # "num_sgd_iter": 45,
            "num_gpus": 1 if device == 'cuda' else 0,
            "num_workers": num_workers,
            "num_cpus_per_worker": 1,
            "num_envs_per_worker": num_envs_per_worker,
            # "lr": 5e-5,
            # "gamma": 0.99,
            "use_gae": True,
            "use_critic": True,
            "batch_mode": "complete_episodes",
            "model": {
                "custom_model": "policy_net",
                "custom_model_config": {
                    "core_hidden_dim": 256,
                    "head_hidden_dim": 32,
                    "wandb_grouping": wandb_name,
                },
            },
            # "env_config": {
            #     "device": "cpu",
            #     "num_envs": num_envs_per_worker,
            #     "scenario_name": scenario_name,
            #     "continuous_actions": True,
            #     "max_steps": max_steps,
            #     "share_reward": True,
            #     # # Scenario specific variables
            #     # "scenario_config": {
            #     #     "n_agents": SCENARIO_CONFIG[scenario_name]["num_agents"],
            #     # },
            # },
            "evaluation_interval": 10,
            "evaluation_duration": 1,
            "evaluation_num_workers": 1,
            "evaluation_parallel_to_training": True,
            # "evaluation_config": {
            #     "num_envs_per_worker": 1,
            #     "env_config": {
            #         "num_envs": 1,
            #     },
            #     "callbacks": MultiCallbacks(callbacks),  # Removed RenderingCallbacks
            # },
            # "callbacks": EvaluationCallbacks,
        },
    )
    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Train policy with SAE')

    # Required
    parser.add_argument('-c', '--scenario', default=None, help='VMAS scenario')
    parser.add_argument('--use_stack', action="store_true", default=False, help='use a stack')

    # Joint observations with encoder
    parser.add_argument('--encoder', default=None, help='Encoder type: mlp/sae. Do not use this option for None')
    parser.add_argument('--encoding_dim', default=None, type=int, help='Encoding dimension')
    parser.add_argument('--encoder_loss', default=None, help='Train encoder loss: policy/recon leave None for frozen')
    parser.add_argument('--encoder_file', default=None, help='File with encoder weights')

    # Misc.
    parser.add_argument('--use_proj', action="store_true", default=False, help='project observations into higher space')
    parser.add_argument('--no_stand', action="store_true", default=False, help='do not standardise observations')
    parser.add_argument('--share_params', action="store_true", default=False, help='share network parameters')

    # Optional
    parser.add_argument('--render', action="store_true", default=False, help='Render environment')
    parser.add_argument('--train_batch_size', default=2560 * 10, type=int, help='train batch size')
    parser.add_argument('--sgd_minibatch_size', default=256, type=int, help='sgd minibatch size')
    parser.add_argument('--training_iterations', default=5000, type=int, help='number of training iterations')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--wandb_name', default="rllib_training", help='wandb run name')

    parser.add_argument('--num_envs', default=32, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--num_cpus_per_worker', default=1, type=int)
    parser.add_argument('-d', '--device', default='cuda')
    args = parser.parse_args()

    train(
        scenario_name=args.scenario,
        use_stack=args.use_stack,
        encoder=args.encoder,
        encoding_dim=args.encoding_dim,
        encoder_file=args.encoder_file,
        encoder_loss=args.encoder_loss,
        use_proj=args.use_proj,
        no_stand=args.no_stand,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.sgd_minibatch_size,
        # max_steps=SCENARIO_CONFIG[args.scenario]["max_steps"],
        training_iterations=args.training_iterations,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        render_env=args.render,
        wandb_name=args.wandb_name,
        share_parameters=args.share_params,
        seed=args.seed,
        device=args.device,
    )