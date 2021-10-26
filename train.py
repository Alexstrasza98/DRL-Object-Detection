#!/usr/bin/env python
import os
import time
import argparse
from datetime import timedelta

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback

np.seterr(under='ignore')

import torch
torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)

from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from utils.env import bbox_env
from utils.tools import *
from utils.models import *
from utils.dataset import read_voc_dataset

# from utils import clr, inv_clr

N_ENV      = 10
N_ROLLOUTS = 100
N_ROLLOUT_STEPS = 200

START_TIME = time.strftime(f'%b%d__%H_%M')
SAVE_ROOT  = os.path.join('models', START_TIME)

def create_env(env_class, image_loader, feature_extractor, fe_out_dim, n_envs):
    env = get_vectorized_env(env_class, image_loader, feature_extractor, fe_out_dim, n_envs)
    env = VecNormalize(env)
    return env

def create_envs(train_loader, test_loader, feature_extractor, fe_out_dim):
    eval_env = create_env( bbox_env, test_loader, feature_extractor, fe_out_dim, n_envs=1 )
    check_env( unwrap_vec_env(eval_env) )
    train_env = create_env( bbox_env, train_loader, feature_extractor, fe_out_dim, n_envs=N_ENV )
    return train_env, eval_env

def fit_model( class_name, train_env, params={}, eval_env=None ):
    model = PPO(
            'MlpPolicy',
            train_env,
            n_epochs = 5,
            n_steps  = N_ROLLOUT_STEPS,
            verbose  = 1,
            # device = 'cpu',
            batch_size = N_ROLLOUT_STEPS * train_env.num_envs,
            **params,
            tensorboard_log = "./tensorboard_log/",
        )
    eval_callback = EvalCallback(
                            eval_env,
                            best_model_save_path=f'{SAVE_ROOT}/{class_name}',
                            log_path='./logs/',
                            eval_freq=N_ROLLOUT_STEPS,
                            n_eval_episodes=10,
                            deterministic=True,
                            render=False
                    )
    model.learn(
        callback = eval_callback,
        total_timesteps = N_ROLLOUTS * N_ROLLOUT_STEPS * train_env.num_envs,
        tb_log_name = f'{class_name}_{START_TIME}'
    )
    return model

def save_model(class_name, model):
    cur_datetime = time.strftime(f'%b%d__%H_%M')
    f_name = os.path.join(SAVE_ROOT,class_name,cur_datetime)
    model.save(f_name+'.zip')
    venv = model.get_vec_normalize_env()
    if venv is not None:
        venv.save(f_name+'.pkl')

def load_model(env, model_path, stats_path):
    model = PPO.load(model_path)
    env = VecNormalize.load(stats_path, env)
    return model, env



def train_class(class_name, train_env, eval_env):
    params = {
        # "n_steps": BATCH_SIZE
    }
    model = fit_model(
                class_name,
                train_env,
                params,
                eval_env=eval_env
            )
    save_model( class_name, model )
    del train_env
    return model

def validate_class(class_name, model, eval_env):
    mean_rew, std_rew = evaluate_policy( model, eval_env, n_eval_episodes=10 )
    print(f'Class {class_name}: mean reward = {mean_rew:.3f} +/- {std_rew:.3f}')


def main():

    train_loader,val_loader = read_voc_dataset(
                                path=f"{VOC2007_ROOT}/VOCtrainval_06-Nov-2007",
                                year='2007'
                              )
    dsets_per_cls_train = sort_class_extract([train_loader])
    dsets_per_cls_val   = sort_class_extract([val_loader])

    feature_extr, feature_extr_dim = get_feature_extractor(use_cuda)

    start_time = time.time()

    for i in range(len(classes)):
        class_name = classes[i]
        print(f"Training class : {class_name} ...")
        train_env, eval_env = create_envs(
                                dsets_per_cls_train[class_name],
                                dsets_per_cls_val[class_name],
                                feature_extr,
                                feature_extr_dim
                                )
        model = train_class( class_name, train_env, eval_env )
        validate_class( class_name, model, eval_env )
        del model
        del eval_env
        torch.cuda.empty_cache()

    print(f"TOTAL TIME : {timedelta(seconds=time.time() - start_time)}" )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--rollouts', type=int, default=N_ROLLOUTS)
    args = parser.parse_args()
    N_ROLLOUTS = args.rollouts
    main()