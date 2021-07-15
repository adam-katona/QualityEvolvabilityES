






import sys
import numpy as np
import math
import copy
import random
import time

import os
import signal
import datetime
import subprocess

import json
from pydoc import locate
import shutil


from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

from distributed_evolution import populations
from distributed_evolution.gen_var_es import GenESOptimizer
from distributed_evolution.gen_var_es import GenVarESOptimizer
from distributed_evolution.gen_ent_es import GenEntESOptimizer
from distributed_evolution.gen_nondominated_es import GenNonDominatedSortESOptimizer
from distributed_evolution.niches import TorchGymNiche
from distributed_evolution.normalizers import (
    centered_rank_normalizer,
    normal_normalizer,
)
#from distributed_evolution.policies.cheetah import CheetahPolicy
from distributed_evolution.policies.my_torch_policies import CheetahPolicyTorch
from distributed_evolution.policies.my_torch_policies import Embedded_2D_CPPN_Net_Policy
from distributed_evolution.policies.my_torch_policies import CAIndirectNet

import torch



def main(
    client, # dask client
    
    ES_type,
    ModelType,
    ModelExtraArgs,

    learning_rate,
    batch_size,
    pop_size,
    l2_coeff,
    noise_std,
    noise_decay,
    n_iterations,
    n_rollouts,
    env_deterministic,
    returns_normalization,
    evals_per_step,
    eval_batch_size,
    single_threaded,
    gpu_mem_frac,
    action_noise,
    progress,
    seed,
    niche_path,
    pop_path,
    po_path,
    device,
    log_fnm,
    env_name,
    extra_config):

    def make_env():
        from distributed_evolution.envs import CheetahEnv, AntEnv
        from distributed_evolution.envs.ant import DirectionalAnt
        from distributed_evolution.envs.humanoid import NoveltyHumanoid

        local_seed = None
        if env_deterministic:
            local_seed = seed

        if env_name.lower() == "cheetah":
            env = CheetahEnv(seed=local_seed,environment_mode="NORMAL")
        elif env_name.lower() == "ant":
            env = AntEnv(seed=local_seed)
        elif env_name.lower() == "directionalant":
            env = DirectionalAnt(seed=local_seed)
        elif env_name.lower() == "deceptiveant":
            env = AntEnv(seed=local_seed,environment_mode="DECEPTIVE")
        elif env_name.lower() == "humanoid":
            env = NoveltyHumanoid(environment_mode="NORMAL",use_built_in_rewards=extra_config["HUMANOID_USE_BUILT_IN_REWARDS"],seed=local_seed)
        elif env_name.lower() == "directionalhumanoid":
            env = NoveltyHumanoid(environment_mode="DIRECTIONAL",use_built_in_rewards=extra_config["HUMANOID_USE_BUILT_IN_REWARDS"],seed=local_seed)
        elif env_name.lower() == "deceptivehumanoid":
            env = NoveltyHumanoid(environment_mode="DECEPTIVE",use_built_in_rewards=extra_config["HUMANOID_USE_BUILT_IN_REWARDS"],seed=local_seed)

        else:
            raise Exception("Invalid env_name")

        return env

    env = make_env()

    if ModelType == "plain_FF":
        model_class = CheetahPolicyTorch   # the original CheetahPolicy is tensorflow
    elif ModelType == "CPPN":
        model_class = Embedded_2D_CPPN_Net_Policy
    elif ModelType == "CA":
        model_class = CAIndirectNet
    else:
        raise "unkown model type!!!"


    model = model_class(   # DEBUG was CheetahPolicy
        env.observation_space.shape,
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        ac_noise_std=action_noise,
        seed=seed,
        gpu_mem_frac=gpu_mem_frac,
        single_threaded=single_threaded,
        extra_args = ModelExtraArgs,
    )

    continue_config = None
    if "continue_config" in extra_config:
        continue_config = extra_config["continue_config"] 
        print("Found continue_config")

    # It is a new run, init niche and population
    if continue_config is None:
        niche = TorchGymNiche(
            make_env, model, ts_limit=1000, n_train_rollouts=n_rollouts, n_eval_rollouts=1
        )

        population = populations.Normal(
            model.get_theta(), noise_std, sigma_decay=noise_decay, device=device
        )

        start_iteration = 0
        target_iterations = n_iterations

    # we are continuing an exisiting unfinished run
    else:
        population = populations.Normal.load(continue_config["pop_filename"])
        niche = TorchGymNiche.load(continue_config["niche_filename"])
        # the way it works is that we first make a checkpoint, and then run a step.
        start_iteration = continue_config["last_iter"] # we want to run the last iter , because it was not yet run, tha chekcpont was seved before the step.
        target_iterations = 800 

        print("loaded checkpoint, ",start_iteration)



    optim = torch.optim.Adam(
        population.parameters(), lr=learning_rate, weight_decay=l2_coeff
    )

    if returns_normalization == "centered_ranks":
        returns_normalizer = centered_rank_normalizer
    elif returns_normalization == "normal":
        returns_normalizer = normal_normalizer
    else:
        raise ValueError("Invalid returns normalizer {}".format(returns_normalizer))


    if ES_type == "plain_ES":   
        es_class = GenESOptimizer
    elif ES_type == "maxvar_ES":
        es_class = GenVarESOptimizer
    elif ES_type == "maxent_ES":
        es_class = GenEntESOptimizer
    elif ES_type == "nondominated_ES":
        es_class = GenNonDominatedSortESOptimizer
    else:
        raise "unknown ES type"



    optimizer = es_class(   # was GenVarESOptimizer
        client,
        population,
        optim,
        niche,
        batch_size=batch_size,
        pop_size=pop_size,
        eval_batch_size=eval_batch_size,
        evals_per_step=evals_per_step,
        returns_normalizer=returns_normalizer,
        seed=seed,
        device=device,
        log_fnm=log_fnm,
    )

    optimizer.optimize(
        n_iterations=target_iterations,
        show_progress=progress,
        pop_path=pop_path,
        niche_path=niche_path,
        po_path=po_path,
        start_iteration=start_iteration
    )




















