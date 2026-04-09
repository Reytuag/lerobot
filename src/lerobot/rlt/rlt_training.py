#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""
import os
import wandb
import concurrent.futures as cf
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import  deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    check_env_attributes_and_types,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)
from lerobot.rlt.autoencoder import Autoencoder

def polyak_update(main_net, target_net, tau=0.005):
    with torch.no_grad():
        for param, target_param in zip(main_net.parameters(), target_net.parameters()):
            target_param.data.lerp_(param.data, tau)


@parser.wrap()
def train_rlt(cfg: EvalPipelineConfig):
    C=10
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    beta_env = os.environ.get("RLT_BETA")
    if beta_env is None:
        beta = 10.0
    else:
        try:
            beta = float(beta_env)
        except ValueError as err:
            raise ValueError(f"Invalid RLT_BETA value: {beta_env}") from err

    action_noise_env = os.environ.get("RLT_ACTION_NOISE")
    if action_noise_env is None:
        action_noise = 0.05
    else:
        try:
            action_noise = float(action_noise_env)
        except ValueError as err:
            raise ValueError(f"Invalid RLT_ACTION_NOISE value: {action_noise_env}") from err

    # optional: number of environment steps over which action noise decays to 0
    action_noise_decay_env = os.environ.get("RLT_ACTION_NOISE_DECAY_STEPS")
    if action_noise_decay_env is None:
        action_noise_decay_steps = None
    else:
        try:
            action_noise_decay_steps = int(action_noise_decay_env)
        except ValueError as err:
            raise ValueError(f"Invalid RLT_ACTION_NOISE_DECAY_STEPS value: {action_noise_decay_env}") from err

    logging.info(colored("RLT beta:", "yellow", attrs=["bold"]) + f" {beta}")
    logging.info(colored("RLT action noise:", "yellow", attrs=["bold"]) + f" {action_noise}")

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Making environment.")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=False,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")
    print("\n"*5)
    print("policy config:")
    print(cfg.policy)
    print("\n"*5)
    cfg.policy.n_action_steps = C
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    print(envs)
    env=envs["libero_spatial"][4]
    seeds=1

    def prepare_obs(observation):
        observation = preprocess_observation(observation)

        # Infer "task" from attributes of environments.
        # TODO: works with SyncVectorEnv but not AsyncVectorEnv
        observation = add_envs_task(env, observation)

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        return observation



 

    policy.reset()
    policy.eval()
    observation, info = env.reset(seed=seeds)

    dataset = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    total_steps = 45000
    # helper: compute current action noise (linear decay over env steps)
    def _current_action_noise(step_idx: int) -> float:
        # step_idx is the outer loop step (each corresponds to C env steps)
        if action_noise <= 0.0:
            return 0.0
        if action_noise_decay_steps is None:
            decay_steps = total_steps
        else:
            decay_steps = max(1, action_noise_decay_steps)
        env_steps = step_idx * C
        progress = min(max(env_steps / decay_steps, 0.0), 1.0)
        return float(action_noise * (1.0 - progress))
    check_env_attributes_and_types(env)


    observation = prepare_obs(observation)

    print(observation)
    with torch.inference_mode():
            batch=observation
            batch = policy._prepare_batch(batch)
            embed,action_flow = policy._get_action_chunk(batch)
    warmup_steps=56
    with torch.no_grad():
        from lerobot.rlt.autoencoder import Autoencoder
        autoencoder=Autoencoder("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",expert_width_multiplier=1,num_expert_layers=4)
        autoencoder.load_state_dict(torch.load("/home/reytuag/VLA/recording_lerobot/lerobot/outputs/eval/2026-03-30/22-40-37_libero_smolvla/autoencoder.pth", weights_only=True))
        autoencoder.eval()
        autoencoder.to(device)
        rl_embed=autoencoder.encode(embed)
        print(rl_embed.shape)
        print(observation["observation.state"].shape)

        from lerobot.rlt.pi_q_networks import Policy_network, Q_network

        input_state=torch.cat([rl_embed,observation["observation.state"]],dim=-1)
        policy_adapt=Policy_network(input_dim=input_state.shape[-1]+action_flow.shape[-1]*C,output_dim=action_flow.shape[-1]*C)
        q_network_1=Q_network(input_dim=input_state.shape[-1]+action_flow.shape[-1]*C,output_dim=1)
        q_network_2=Q_network(input_dim=input_state.shape[-1]+action_flow.shape[-1]*C,output_dim=1)
        q_network_target_1=deepcopy(q_network_1)
        q_network_target_2=deepcopy(q_network_2)
        policy_adapt.to(device)
        q_network_1.to(device)
        q_network_2.to(device)
        q_network_target_1.to(device)
        q_network_target_2.to(device)
        if step < warmup_steps:
            action_adapt = action_flow[:, :C]
        else:
            action_adapt = policy_adapt.forward(input_state, action_flow[:, :C])
            action_adapt = action_adapt.view(action_flow[:, :C].shape)
            # add gaussian noise to action adapt for exploration with clipping
            cur_noise = _current_action_noise(step)
            if cur_noise > 0.0:
                sampled = torch.randn_like(action_adapt) * cur_noise
                sampled = torch.clamp(sampled, -2.0 * cur_noise, 2.0 * cur_noise)
                action_adapt = action_adapt + sampled


    optimizer_policy_adapt=torch.optim.Adam(policy_adapt.parameters(),lr=1e-4)
    optimizer_q_network_1=torch.optim.Adam(q_network_1.parameters(),lr=1e-4)
    optimizer_q_network_2=torch.optim.Adam(q_network_2.parameters(),lr=1e-4)
    wandb.init(
        project="lerobot-autoencoder",
        name=f"target_net_beta_{beta}_noise_{action_noise}_{int(time.time())}"
    )

    steps_current=np.zeros(env.num_envs)
    nb_successes=0
    nb_episodes=0
    avg_steps_success=0
    avg_steps_episode=0
    for step in trange(
        total_steps//C,
        desc=f"Running rollout with at most {total_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    ):

        # print(action_flow.shape)
        
        list_reward = []
        done = np.array([False] * env.num_envs)

    
        for i in range(C):
            #replace dones with dummy actions
            
        
            action = postprocessor(action_adapt[:,i])
            action_transition = {ACTION: action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition[ACTION]

            # Convert to CPU / numpy.
            action_numpy: np.ndarray = action.to("cpu").numpy()

            dummy_action=np.array([[0, 0, 0, 0, 0, 0, -1]])
            dummy_action=dummy_action.repeat(env.num_envs,0)
            action_numpy[done] = dummy_action[done]
            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            list_reward.append(reward)
            
            steps_current += 1- done.astype(int)
            # print(reward)
            # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
            # available if none of the envs finished.
            successes=reward > 0
            if(successes.any()):
                nb_successes+=successes.sum()
                print("==================successes=================")
                print(successes)

                avg_steps_success+=(steps_current*successes.astype(int)).sum()

            done = terminated | truncated | done
        nb_episodes+=done.sum()

        if done.any():
            avg_steps_episode+=(steps_current*done.astype(int)).sum()


            steps_current=steps_current*(1-done)


        observation = prepare_obs(observation)
 
        with torch.no_grad():
            policy_adapt.eval()
            batch=observation
            batch = policy._prepare_batch(batch)
            embed,new_action_flow = policy._get_action_chunk(batch)
            rl_embed=autoencoder.encode(embed)
            new_input_state=torch.cat([rl_embed,observation["observation.state"]],dim=-1)
            def _current_action_noise(step_idx: int) -> float:
                # step_idx is the outer loop step (each corresponds to C env steps)
                # compute decay over env steps: env_steps = step_idx * C
                if action_noise <= 0.0:
                    return 0.0
                if action_noise_decay_steps is None:
                    decay_steps = total_steps
                else:
                    decay_steps = max(1, action_noise_decay_steps)
                env_steps = step_idx * C
                progress = min(max(env_steps / decay_steps, 0.0), 1.0)
                return float(action_noise * (1.0 - progress))

            if step < warmup_steps:
                new_action_adapt = new_action_flow[:, :C]
            else:
                new_action_adapt = policy_adapt.forward(input_state, new_action_flow[:, :C])
                new_action_adapt = new_action_adapt.view(new_action_flow[:, :C].shape)
                # add gaussian noise to action adapt for exploration with clipping
                cur_noise = _current_action_noise(step)
                if cur_noise > 0.0:
                    sampled = torch.randn_like(new_action_adapt) * cur_noise
                    sampled = torch.clamp(sampled, -2.0 * cur_noise, 2.0 * cur_noise)
                    new_action_adapt =new_action_adapt + sampled
                

                 #
            discounted_reward = 0
            gamma = 0.99
            # we don't care about done bc if done is true the env is reset and we fill with dummy actions until next 
            for i in reversed(range(C)):
                discounted_reward = list_reward[i] + gamma * discounted_reward

            for i in range(env.num_envs):
                dataset.append((input_state[i].cpu().numpy(),action_flow[i,:C].cpu().numpy(),action_adapt[i].cpu().numpy(),new_input_state[i].cpu().numpy(),new_action_flow[i,:C].cpu().numpy(),discounted_reward[i],done[i]))

            action_flow = new_action_flow
            action_adapt = new_action_adapt
            input_state = new_input_state

        if(step+1) % 10 == 0:
            print(f'step: {step}, action_noise: {_current_action_noise(step):.4f}, success rate: {nb_successes/nb_episodes:.4f}')
        if(step+1) % 56 == 0:
            print(f"success rate: {nb_successes/nb_episodes}")
            wandb.log({"success_rate": nb_successes/nb_episodes,"throughput": nb_successes,"avg_steps_success": avg_steps_success/nb_successes if nb_successes > 0 else 0,"avg_steps_episode": avg_steps_episode/nb_episodes if nb_episodes > 0 else 0})
            nb_successes=0
            nb_episodes=0
            avg_steps_episode=0
            avg_steps_success=0
            # save dataset

        # training part
        

        batch_size=min(1024,len(dataset))
        if( step > warmup_steps):
            for batch_step in range(5):
                
                # sample from dataset
                batch_samples = np.random.choice(len(dataset), batch_size, replace=False)
                batch_input_state = torch.tensor(np.array([dataset[i][0] for i in batch_samples]), dtype=torch.float32).to(device)
                batch_action_flow = torch.tensor(np.array([dataset[i][1] for i in batch_samples]), dtype=torch.float32).to(device)
                batch_action_adapt = torch.tensor(np.array([dataset[i][2] for i in batch_samples]), dtype=torch.float32).to(device)
                batch_new_input_state = torch.tensor(np.array([dataset[i][3] for i in batch_samples]), dtype=torch.float32).to(device)
                batch_new_action_flow = torch.tensor(np.array([dataset[i][4] for i in batch_samples]), dtype=torch.float32).to(device)  
                batch_discounted_reward = torch.tensor(np.array([dataset[i][5] for i in batch_samples]), dtype=torch.float32).to(device)
                batch_done = torch.tensor(np.array([dataset[i][6] for i in batch_samples]), dtype=torch.float32).to(device) 
                
                # compute target q value
                with torch.no_grad():
                    policy_adapt.eval()
                    q_network_1.eval()
                    q_network_2.eval()
                    sample_new_action_adapt = policy_adapt.forward(batch_new_input_state,batch_new_action_flow[:,:C])
                    sample_new_action_adapt = sample_new_action_adapt.view(batch_new_action_flow[:,:C].shape)
                    target_q_value_1 = q_network_target_1.forward(batch_new_input_state,sample_new_action_adapt)
                    target_q_value_2 = q_network_target_2.forward(batch_new_input_state,sample_new_action_adapt)
                    target_q_value = torch.min(target_q_value_1,target_q_value_2)
                    # print(batch_new_input_state.shape)
                    # print(target_q_value.shape)
                    # print(batch_discounted_reward.shape)
                    # print(batch_done.shape)
                    target_q_value = batch_discounted_reward + 0.99**C * target_q_value * (1 - batch_done)
                
                policy_adapt.train()
                q_network_1.train()
                q_network_2.train()
                # compute current q value
                current_q_value_1 = q_network_1.forward(batch_input_state,batch_action_adapt)
                current_q_value_2 = q_network_2.forward(batch_input_state,batch_action_adapt)
                # compute critic loss
                critic_loss_1 = nn.MSELoss()(current_q_value_1,target_q_value)
                critic_loss_2 = nn.MSELoss()(current_q_value_2,target_q_value)
                # update critic
                optimizer_q_network_1.zero_grad()
                critic_loss_1.backward()
                optimizer_q_network_1.step()

                optimizer_q_network_2.zero_grad()
                critic_loss_2.backward()
                optimizer_q_network_2.step()

                if(step%3 == 0):
                    # compute actor loss
                    # random mask on action flow
                    mask = torch.rand(batch_size) > -0.1
                    batch_action_flow_mask = batch_action_flow[:,:C] * mask[:,None,None].to(device)

                    training_action_adapt = policy_adapt.forward(batch_input_state,batch_action_flow_mask)
                    training_action_adapt = training_action_adapt.view(batch_action_flow[:,:C].shape)

                    actor_qloss= -q_network_1.forward(batch_input_state,training_action_adapt).mean()
                    actor_bcloss=nn.MSELoss()(training_action_adapt,batch_action_flow[:,:C])
                    actor_loss = actor_qloss+beta*actor_bcloss
                    # update actor
                    optimizer_policy_adapt.zero_grad()
                    actor_loss.backward()
                    optimizer_policy_adapt.step()


                    polyak_update(q_network_1, q_network_target_1, tau=0.02)
                    polyak_update(q_network_2, q_network_target_2, tau=0.02)
                    # print(f"step: {step}, critic_loss_1: {critic_loss_1.item()}, critic_loss_2: {critic_loss_2.item()}, actor_loss: {actor_loss.item()}")
                    wandb.log({"critic_loss_1": critic_loss_1.item(), "critic_loss_2": critic_loss_2.item(), "actor_loss": actor_loss.item(),"actor_qloss": actor_qloss.item(), "actor_bcloss": actor_bcloss.item()})  
    

        


    
    # Close all vec envs
    close_envs(envs)

    logging.info("End of training")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")







def main():
    init_logging()
    register_third_party_plugins()
    train_rlt()


if __name__ == "__main__":
    main()
