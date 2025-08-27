import os
import random
from datetime import datetime
import time
from pathlib import Path
import threading
from queue import Queue

import pandas as pd

from ssh_handler import SSHConnection
import Stopping_Rules, data_transfer_wrapper
from Helpers import *
from NestedSSHHandler import NestedSSHClient
from Web_Wrapper import *
from cost_model_config_finder import get_next_suggestion, get_first_suggestion, \
    get_next_suggestion_2_phase_search, get_next_suggestion_2_phase_search_batch, calculate_search_space_size
from environments import *
from model_implementations.Weighted_Combination_RF_Cost_Model import \
    Per_Environment_RF_Cost_Model
from model_implementations.lhs_search_optimizer import LHS_Search_Optimizer
from model_implementations.openbox_ask_tell import OpenBox_Ask_Tell
from model_implementations.syne_tune_ask_tell import Syne_Tune_Ask_Tell

def main():
    all_envs()

    return

    ssh_host = "sr630-wn-a-20"

    environment = env_S8_C8_N1000
    #'''
    cost_model = init_cost_model(environment)

    ssh = NestedSSHClient(jump_host=big_cluster_main_host,
                          jump_username=get_username_for_host(big_cluster_main_host),
                          target_host=ssh_host,
                          target_username=get_username_for_host(ssh_host))

    online_optimization_loop_cost_model(cost_model=cost_model,
                             environment=environment,
                             metric='uncompressed_throughput',
                             mode="max",
                             config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                             max_queries_per_iteration=1000,
                             reconfig_intervall_seconds=5,
                             ssh=ssh)
    #'''
    #'''
    online_optimization_loop_bayesian(environment=environment,
                                        metric='uncompressed_throughput',
                                        mode="max",
                                        config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                                        reconfig_intervall_seconds=5,
                                        ssh=ssh)
    #'''
    #'''
    online_optimization_loop_rgpe(environment=environment,
                                      metric='uncompressed_throughput',
                                      mode="max",
                                      config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                                      reconfig_intervall_seconds=5,
                                      ssh=ssh)
    #'''


def all_envs():
    ssh_host = "sr630-wn-a-20"
    for environment in [
        #env_S2_C2_N50,
        #env_S2_C8_N50,
        #env_S2_C16_N50,

        #env_S8_C2_N150,
        #env_S8_C8_N150,
        #env_S8_C16_N150,

        #env_S16_C2_N1000,
        #env_S16_C8_N1000,
        env_S16_C16_N1000,
    ]:
        try:
            cost_model = init_cost_model(environment)

            ssh = NestedSSHClient(jump_host=big_cluster_main_host,
                                  jump_username=get_username_for_host(big_cluster_main_host),
                                  target_host=ssh_host,
                                  target_username=get_username_for_host(ssh_host))
           # '''
            try:
                online_optimization_loop_cost_model(cost_model=cost_model,
                                                environment=environment,
                                                metric='uncompressed_throughput',
                                                mode="max",
                                                config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                                                max_queries_per_iteration=1000,
                                                reconfig_intervall_seconds=10,
                                                ssh=ssh)
            except:
                pass
            #'''
            #'''
            try:
                online_optimization_loop_bayesian(environment=environment,
                                              metric='uncompressed_throughput',
                                              mode="max",
                                              config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                                              reconfig_intervall_seconds=10,
                                              ssh=ssh)
            except:
                pass
            #'''
            #'''
            try:
                online_optimization_loop_random(environment=environment,
                                                  metric='uncompressed_throughput',
                                                  mode="max",
                                                  config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                                                  reconfig_intervall_seconds=10,
                                                  ssh=ssh)
            except:
                pass
            #'''
            #'''
            try:
                online_optimization_loop_rgpe(environment=environment,
                                          metric='uncompressed_throughput',
                                          mode="max",
                                          config_space=config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz,
                                          reconfig_intervall_seconds=10,
                                          ssh=ssh)
            except:
                pass
                
            #'''
        except:
            pass


def init_cost_model(environment,metric = "uncompressed_throughput",training_data_per_env=400,algorithm="cost_model_vx01",regression_model="xgb",network_transformation=True):
    input_fields = [
        "client_bufpool_factor",
        "server_bufpool_factor",
        "buffer_size",
        "compression",
        "send_par",
        "write_par",
        "decomp_par",
        "read_par",
        "deser_par",
        "ser_par",
        "comp_par"
    ]

    cost_model = Per_Environment_RF_Cost_Model(input_fields=input_fields,
                                               metric=metric,
                                               data_per_env=training_data_per_env,
                                               underlying=algorithm,
                                               cluster=cluster,
                                               regression_model=regression_model,
                                               network_transformation=network_transformation,
                                               history_ratio=0.8#history_ratio dynamic ration currently
                                               )

    data, suffix = get_transfer_learning_data_for_environment(environment, use_all_environments=False, data_per_env=training_data_per_env)

    x = data#[input_fields]
    y = data[metric]

    cost_model.train(x, y)

    return cost_model


def get_transfer_learning_data_for_environment(target_environment, use_all_environments, data_per_env, except_N_most_similar=1):

    base_path = "random_samples_10_5M_semi_flex"
    environments_to_use = []
    suffix = ""



    dict_environment_most_similars = {
        "S2_C2_N50": ['S2_C2_N50', 'S2_C2_N100000', 'S2_C2_N1000', 'S2_C2_N150', 'S8_C2_N50', 'S16_C2_N50', 'S8_C2_N100000', 'S16_C2_N1000', 'S8_C2_N1000', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N150', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C2_N150": ['S2_C2_N150', 'S2_C2_N1000', 'S8_C2_N100000', 'S2_C2_N100000', 'S16_C2_N1000', 'S8_C2_N1000', 'S16_C2_N100000', 'S16_C2_N150', 'S8_C2_N50', 'S16_C2_N50', 'S2_C2_N50', 'S8_C2_N150', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C2_N1000": ['S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N150', 'S8_C2_N100000', 'S16_C2_N1000', 'S8_C2_N1000', 'S8_C2_N50', 'S2_C2_N50', 'S16_C2_N50', 'S16_C2_N100000', 'S16_C2_N150', 'S8_C2_N150', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C2_N100000": ['S2_C2_N100000', 'S2_C2_N1000', 'S2_C2_N150', 'S2_C2_N50', 'S8_C2_N100000', 'S16_C2_N1000', 'S8_C2_N1000', 'S8_C2_N50', 'S16_C2_N50', 'S16_C2_N100000', 'S16_C2_N150', 'S8_C2_N150', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C8_N50": ['S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S8_C16_N50', 'S2_C2_N1000', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C8_N150": ['S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C16_N50', 'S8_C8_N50', 'S16_C16_N50', 'S16_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C8_N1000": ['S2_C8_N1000', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C16_N150', 'S2_C8_N100000', 'S2_C8_N150', 'S2_C16_N50', 'S2_C8_N50', 'S8_C16_N50', 'S8_C8_N50', 'S16_C16_N50', 'S16_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C8_N100000": ['S2_C8_N100000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C8_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S2_C16_N50', 'S2_C8_N50', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C16_N50": ['S2_C16_N50', 'S2_C8_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S2_C2_N150', 'S16_C8_N50', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C16_N150": ['S2_C16_N150', 'S2_C8_N150', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C16_N50', 'S8_C8_N50', 'S16_C16_N50', 'S16_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C16_N1000": ['S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C16_N50', 'S8_C8_N50', 'S16_C16_N50', 'S16_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S2_C16_N100000": ['S2_C16_N100000', 'S2_C8_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N1000', 'S2_C8_N150', 'S2_C16_N50', 'S2_C8_N50', 'S8_C16_N50', 'S8_C8_N50', 'S16_C16_N50', 'S16_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S8_C2_N50": ['S8_C2_N50', 'S16_C2_N50', 'S16_C2_N150', 'S8_C2_N1000', 'S8_C2_N100000', 'S16_C2_N1000', 'S2_C2_N150', 'S2_C2_N1000', 'S8_C2_N150', 'S16_C2_N100000', 'S2_C2_N50', 'S2_C2_N100000', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S8_C2_N150": ['S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S8_C2_N50', 'S16_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S8_C2_N1000": ['S8_C2_N1000', 'S8_C2_N100000', 'S16_C2_N1000', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N150', 'S2_C2_N150', 'S8_C2_N50', 'S2_C2_N1000', 'S16_C2_N50', 'S2_C2_N100000', 'S2_C2_N50', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S8_C2_N100000": ['S8_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S16_C2_N100000', 'S16_C2_N150', 'S2_C2_N150', 'S8_C2_N150', 'S2_C2_N1000', 'S8_C2_N50', 'S16_C2_N50', 'S2_C2_N100000', 'S2_C2_N50', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S8_C8_N50": ['S8_C8_N50', 'S16_C16_N50', 'S16_C8_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N100000', 'S2_C8_N100000', 'S2_C16_N1000', 'S2_C8_N1000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S8_C2_N150', 'S16_C16_N1000', 'S16_C2_N150', 'S16_C2_N100000', 'S16_C16_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C8_N150": ['S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S8_C8_N100000', 'S16_C8_N1000', 'S16_C8_N100000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C8_N1000": ['S8_C8_N1000', 'S8_C8_N100000', 'S16_C8_N1000', 'S16_C8_N100000', 'S8_C16_N1000', 'S16_C8_N150', 'S16_C16_N150', 'S8_C16_N100000', 'S8_C16_N150', 'S16_C16_N1000', 'S8_C8_N150', 'S16_C16_N100000', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C8_N100000": ['S8_C8_N100000', 'S16_C8_N1000', 'S16_C8_N100000', 'S8_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N150', 'S16_C8_N150', 'S16_C16_N1000', 'S8_C16_N150', 'S16_C16_N100000', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C16_N50": ['S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C8_N100000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C8_N1000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C2_N150', 'S8_C16_N1000', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C16_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S16_C16_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S16_C16_N100000', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C16_N150": ['S8_C16_N150', 'S8_C8_N150', 'S16_C16_N150', 'S16_C8_N150', 'S8_C8_N1000', 'S8_C8_N100000', 'S16_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C16_N1000": ['S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S8_C8_N1000', 'S16_C16_N100000', 'S16_C16_N150', 'S16_C8_N150', 'S8_C16_N150', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S8_C16_N100000": ['S8_C16_N100000', 'S8_C16_N1000', 'S16_C16_N1000', 'S16_C16_N100000', 'S16_C8_N100000', 'S16_C8_N1000', 'S8_C8_N100000', 'S8_C8_N1000', 'S16_C16_N150', 'S16_C8_N150', 'S8_C16_N150', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C2_N50": ['S16_C2_N50', 'S8_C2_N50', 'S16_C2_N150', 'S8_C2_N1000', 'S8_C2_N100000', 'S16_C2_N1000', 'S2_C2_N150', 'S2_C2_N1000', 'S8_C2_N150', 'S16_C2_N100000', 'S2_C2_N50', 'S2_C2_N100000', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S16_C2_N150": ['S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S8_C2_N150', 'S16_C2_N1000', 'S8_C2_N100000', 'S8_C2_N50', 'S16_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S16_C2_N1000": ['S16_C2_N1000', 'S8_C2_N1000', 'S8_C2_N100000', 'S16_C2_N100000', 'S16_C2_N150', 'S8_C2_N150', 'S2_C2_N150', 'S8_C2_N50', 'S2_C2_N1000', 'S16_C2_N50', 'S2_C2_N100000', 'S2_C2_N50', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S16_C2_N100000": ['S16_C2_N100000', 'S16_C2_N150', 'S16_C2_N1000', 'S8_C2_N1000', 'S8_C2_N150', 'S8_C2_N100000', 'S2_C2_N150', 'S8_C2_N50', 'S16_C2_N50', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50', 'S2_C8_N50', 'S2_C16_N50', 'S2_C8_N100000', 'S2_C8_N1000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C16_N150', 'S2_C8_N150', 'S8_C16_N50', 'S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000'],
        "S16_C8_N50": ['S16_C8_N50', 'S16_C16_N50', 'S8_C8_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N100000', 'S2_C8_N100000', 'S2_C16_N1000', 'S2_C8_N1000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S8_C2_N150', 'S16_C16_N100000', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C8_N150": ['S16_C8_N150', 'S16_C16_N150', 'S8_C16_N150', 'S8_C8_N150', 'S8_C8_N1000', 'S8_C8_N100000', 'S16_C8_N1000', 'S16_C8_N100000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C8_N1000": ['S16_C8_N1000', 'S8_C8_N100000', 'S16_C8_N100000', 'S8_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N150', 'S16_C8_N150', 'S16_C16_N1000', 'S8_C16_N150', 'S16_C16_N100000', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C8_N100000": ['S16_C8_N100000', 'S16_C8_N1000', 'S8_C8_N100000', 'S8_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N150', 'S16_C8_N150', 'S16_C16_N1000', 'S8_C16_N150', 'S16_C16_N100000', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C16_N50": ['S16_C16_N50', 'S8_C8_N50', 'S16_C8_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C8_N100000', 'S2_C16_N100000', 'S2_C16_N1000', 'S2_C8_N1000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C8_N150', 'S8_C16_N150', 'S16_C8_N150', 'S16_C16_N150', 'S8_C8_N1000', 'S16_C8_N100000', 'S8_C8_N100000', 'S16_C8_N1000', 'S8_C16_N1000', 'S8_C16_N100000', 'S8_C2_N150', 'S16_C16_N1000', 'S16_C2_N150', 'S16_C2_N100000', 'S16_C16_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C16_N150": ['S16_C16_N150', 'S16_C8_N150', 'S8_C16_N150', 'S8_C8_N150', 'S8_C8_N1000', 'S8_C8_N100000', 'S16_C8_N1000', 'S16_C8_N100000', 'S8_C16_N1000', 'S8_C16_N100000', 'S16_C16_N1000', 'S16_C16_N100000', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C16_N1000": ['S16_C16_N1000', 'S8_C16_N100000', 'S8_C16_N1000', 'S16_C16_N100000', 'S16_C8_N1000', 'S8_C8_N100000', 'S16_C8_N100000', 'S8_C8_N1000', 'S16_C16_N150', 'S16_C8_N150', 'S8_C16_N150', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
        "S16_C16_N100000": ['S16_C16_N100000', 'S8_C16_N100000', 'S16_C16_N1000', 'S8_C16_N1000', 'S16_C8_N100000', 'S16_C8_N1000', 'S8_C8_N100000', 'S8_C8_N1000', 'S16_C16_N150', 'S16_C8_N150', 'S8_C16_N150', 'S8_C8_N150', 'S16_C8_N50', 'S8_C8_N50', 'S16_C16_N50', 'S8_C16_N50', 'S2_C8_N150', 'S2_C16_N150', 'S2_C16_N1000', 'S2_C16_N100000', 'S2_C8_N1000', 'S2_C8_N100000', 'S2_C16_N50', 'S2_C8_N50', 'S8_C2_N150', 'S16_C2_N150', 'S16_C2_N100000', 'S8_C2_N1000', 'S16_C2_N1000', 'S8_C2_N100000', 'S16_C2_N50', 'S8_C2_N50', 'S2_C2_N150', 'S2_C2_N1000', 'S2_C2_N100000', 'S2_C2_N50'],
    }

    if environment_to_string(target_environment) in dict_environment_most_similars.keys():

        environments_to_use = dict_environment_most_similars[environment_to_string(target_environment)][except_N_most_similar:]

        environments_not_used = dict_environment_most_similars[environment_to_string(target_environment)][:except_N_most_similar]

        suffix = "_exc"

        suffix_alt = suffix + f"_top_{except_N_most_similar}_similar"

        for environment in environments_not_used:
            suffix = suffix + "_" + environment.replace("_", "")


    if use_all_environments:

        if target_environment in all_base_environments:
            environments_to_use = all_base_signatures
            suffix = "_all_envs"


    time_threshold = 2
    file_list = [glob.glob(f"{base_path}/{signature}_random_sample*.csv") for signature in environments_to_use]
    data_frames = []



    limit_random_samples = int(data_per_env / 2)
    limit_high = int(data_per_env/4)
    limit_low = int(data_per_env/4)

    if ((limit_random_samples + limit_high + limit_low) - data_per_env) > 1:
        limit_random_samples = limit_random_samples + int((limit_random_samples + limit_high + limit_low) - data_per_env)



    base_path_high = "engineered_data_base/high_data"
    base_path_low = "engineered_data_base/low_data"

    file_list_high = [glob.glob(f"{base_path_high}/{signature}*.csv") for signature in environments_to_use]
    for file in file_list_high:
        if file:
            df = pd.read_csv(file[0])
            df = df[(df['time'] > time_threshold)]
            df = df.head(limit_high)
            data_frames.append(df)

    file_list_low = [glob.glob(f"{base_path_low}/{signature}*.csv") for signature in environments_to_use]
    for file in file_list_low:
        if file:
            df = pd.read_csv(file[0])
            df = df[(df['time'] > time_threshold)]
            df = df.head(limit_low)
            data_frames.append(df)


    for file in file_list:
        if file:
            df = pd.read_csv(file[0])
            df = df[(df['transfer_id'] > 0)]
            df = df.head(limit_random_samples)
            data_frames.append(df)
    data = pd.concat(data_frames, axis=0, ignore_index=True) if data_frames else pd.DataFrame()

    return data, suffix


def map_configuration(config):
    config["rcv_par"] = config["send_par"]


    if "client_bufpool_factor" in config.keys():
        thread_count_client = config['write_par'] + config['ser_par'] + config['decomp_par'] + config['rcv_par']
        min_buffer_count_client = thread_count_client * 2
        config["client_buffpool_size"] = min_buffer_count_client * config["client_bufpool_factor"] * config['buffer_size']

    if "server_bufpool_factor" in config.keys():
        thread_count_server = config['read_par'] + config['deser_par'] + config['comp_par'] + config['send_par']
        min_buffer_count_server = thread_count_server * 2
        config["server_buffpool_size"] = min_buffer_count_server * config["server_bufpool_factor"] * config['buffer_size']

    config['client_buffpool_size'] = config['buffer_size'] * 64 * 4
    config['server_buffpool_size'] = config['buffer_size'] * 64 * 4


    config['send_par'] = 1

    if 'format' in config.keys():
        config['src_format'] = config['format']
        config['target_format'] = config['format']

    if 'compression' in config.keys():
        config['compression_lib'] = config['compression']

    return config


def reduce_search_sapce(config):
    reduced_config_space = []

    #these stay fixed
    reduced_config_space.append({'name': "format", 'type': 'categorical', 'domain': [config["format"]]})
    reduced_config_space.append({'name': "client_bufpool_factor", 'type':'categorical', 'domain': [config["client_bufpool_factor"]]})
    reduced_config_space.append({'name': "server_bufpool_factor", 'type': 'categorical', 'domain': [config["server_bufpool_factor"]]})
    reduced_config_space.append({'name': "buffer_size", 'type': 'categorical', 'domain': [config["buffer_size"]]})
    reduced_config_space.append({'name': "send_par", 'type': 'categorical', 'domain': [config["send_par"]]})

    # the parralelisms can be reconfigured
    reduced_config_space.append({'name': "write_par", 'type': 'integer', 'lower': 1, 'upper': 16})
    reduced_config_space.append({'name': "decomp_par", 'type': 'integer', 'lower': 1, 'upper': 16})
    reduced_config_space.append({'name': "read_par", 'type': 'integer', 'lower': 1, 'upper': 16})
    reduced_config_space.append({'name': "deser_par", 'type': 'integer', 'lower': 1, 'upper': 16})
    reduced_config_space.append({'name': "ser_par", 'type': 'integer', 'lower': 1, 'upper': 16})
    reduced_config_space.append({'name': "comp_par", 'type': 'integer', 'lower': 1, 'upper': 16})

    # compression can also be reconfigured
    reduced_config_space.append({'name': "compression", 'type': 'categorical', 'domain': ["nocomp", "zstd", "lz4", "lzo", "snappy"]})

    return reduced_config_space


def start_docker_setup(ssh):
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Starting setting up docker ressources")

    ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml down")
    ssh.execute_cmd("docker stop $(docker ps --filter \"ancestor=xdbc-controller:latest\" -q)")
    ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml up -d")

    cmd = ("bash -lc \" make -C XDBCgamedemo/xdbc-controller start &&  docker exec -d xdbc-controller python3 xdbc_controller.py\"")
    ssh.execute_cmd(cmd,background=False,get_pty=True)

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Finished setting up docker ressources")
def online_optimization_loop_cost_model(cost_model, environment, metric, mode, config_space, max_queries_per_iteration, reconfig_intervall_seconds, ssh):

    config_space_string = get_config_space_string(config_space)
    filename = f"results_{cost_model.underlying}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    filepath = f"results_online3_{config_space_string}/{environment_to_string(environment)}/{cost_model.underlying}/"
    output_file = filepath + filename
    Path(filepath).mkdir(parents=True, exist_ok=True)

    start_docker_setup(ssh)

    # get inintal configuration
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Querying Cost Model for initial configuration for {max_queries_per_iteration} queries")
    inital_configuration = get_next_suggestion_2_phase_search_batch(config_space, max_queries_per_iteration, cost_model, environment, mode, metric)
    # to print weight table
    #cost_model.predict(inital_configuration,environment,True)

    # map config to actual paramter values
    inital_paramters = map_configuration(inital_configuration)

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Starting transfer with initial configuration {inital_configuration}")

    # start transfer
    complete_config = {**inital_paramters, **fixed_parameters}
    start_xdbc_transfer_parameter_wrapper(ssh, environment, complete_config, complete_config )
    reconfiguration_timestamp = datetime.now()
    start_transfer_timestamp = datetime.now()

    # recompute search space to have non-reconfigurable parameters fixed
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reducing search space to keep non reconfigurable parameters fixed")
    reduced_search_space = reduce_search_sapce(inital_configuration)
    search_space_size = calculate_search_space_size(config_space)
    reduced_search_space_size = calculate_search_space_size(reduced_search_space)
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reduced search space has a size of {reduced_search_space_size}, original size was {search_space_size}, reduced to {reduced_search_space_size/search_space_size}")

    transfer_running = True
    current_configuration = inital_configuration
    result = {}

    (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
    previous_percentage = result_perc
    previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

    # add loop that waits till non zero percentage is returned
    no_progress = True
    if previous_percentage != 0:
        no_progress = False

    while(no_progress):
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        previous_percentage = result_perc
        previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

        if previous_percentage != 0:
            no_progress = False
            time.sleep(reconfig_intervall_seconds)
            break
        else:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] Observed zero progress, waiting...")
        time.sleep(1)


    while(transfer_running):

        # get metrics
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        current_metric_poll_timestamp = datetime.fromisoformat(timestamp)
        current_percentage = result_perc

        throughput_since_last = (FILESIZE * (current_percentage - previous_percentage) / 100) / 1_000_000 / ((current_metric_poll_timestamp - previous_metric_poll_timestamp).total_seconds())
        result_metric = throughput_since_last

        # save all datapoints
        result[metric] = result_metric#[metric]
        result['percentage_progress'] = current_percentage
        result['metric_poll_timestamp'] = current_metric_poll_timestamp
        result['reconfiguration_timestamp'] = reconfiguration_timestamp
        result.update(current_configuration)

        previous_metric_poll_timestamp = current_metric_poll_timestamp
        previous_percentage = current_percentage

        df = pd.DataFrame(result, index=[0])
        df.to_csv(filepath + filename, mode='a', header=(not os.path.isfile(output_file)))
        result = {}

        if current_percentage >= 100:
            transfer_running = False
            break

        #update the cost model with the new result
        print(f"[{datetime.today().strftime('%H:%M:%S')}] Updating Cost Model with result {result_metric}, progress at {current_percentage}%")
        cost_model.update(current_configuration, {'uncompressed_throughput':result_metric})

        # query cost model for next parameter configuration
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Querying Cost Model for next configuration for {max_queries_per_iteration} queries")
        current_configuration = get_next_suggestion_2_phase_search_batch(reduced_search_space, max_queries_per_iteration, cost_model, environment, mode, metric)
        current_parameters = map_configuration(current_configuration)
        # to print weight table
        #cost_model.predict(current_configuration, environment,True)

        # reconfigure trasnfer
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfiguring running transfer with new configuration {current_configuration}")
        reconfiguration_timestamp = datetime.now()
        reconfigure_xdbc_transfer_wrapper(ssh,current_parameters,current_parameters)

        # wait intervcall
        time.sleep(reconfig_intervall_seconds)

    end_transfer_timestmap = datetime.now()

    print(f"Transfer completed in {(end_transfer_timestmap - start_transfer_timestamp).total_seconds()} seconds")
    print(f"Average throughput of  {FILESIZE/1000/1000 / (end_transfer_timestmap - start_transfer_timestamp).total_seconds()} MB/s")


#FILESIZE = 78979656654 # sf100
#FILESIZE = 7715741636 # sf10

if fixed_parameters['table'] == 'lineitem_sf10':
    FILESIZE = 7715741636
else:
    FILESIZE = 78979656654


def online_optimization_loop_bayesian(environment, metric, mode, config_space, reconfig_intervall_seconds, ssh):

    config_space_string = get_config_space_string(config_space)
    filename = f"results_bayesian_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    filepath = f"results_3_{config_space_string}/{environment_to_string(environment)}/bayesian/"
    output_file = filepath + filename
    Path(filepath).mkdir(parents=True, exist_ok=True)

    start_docker_setup(ssh)

    optimizer = OpenBox_Ask_Tell(config_space=config_space, metric=metric, mode=mode, underlying='bayesian_open_box')

    # get inintal configuration
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Querying Optimizer for initial configuration")
    inital_configuration = optimizer.suggest()

    # map config to actual paramter values
    inital_paramters = map_configuration(inital_configuration)

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Starting transfer with initial configuration {inital_configuration}")

    # start transfer
    complete_config = {**inital_paramters, **fixed_parameters}
    start_xdbc_transfer_parameter_wrapper(ssh, environment, complete_config, complete_config )
    reconfiguration_timestamp = datetime.now()
    start_transfer_timestamp = datetime.now()

    # recompute search space to have non-reconfigurable parameters fixed
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reducing search space to keep non reconfigurable parameters fixed")
    reduced_search_space = reduce_search_sapce(inital_configuration)
    search_space_size = calculate_search_space_size(config_space)
    reduced_search_space_size = calculate_search_space_size(reduced_search_space)
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reduced search space has a size of {reduced_search_space_size}, original size was {search_space_size}, reduced to {reduced_search_space_size/search_space_size}")

    # restart optimzer with reduced search space
    optimizer = OpenBox_Ask_Tell(config_space=reduced_search_space, metric=metric, mode=mode, underlying='bayesian_open_box')

    transfer_running = True
    current_configuration = inital_configuration
    result = {}

    (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
    previous_percentage = result_perc
    previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

    # add loop that waits till non zero percentage is returned
    no_progress = True
    if previous_percentage != 0:
        no_progress = False

    while(no_progress):
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        previous_percentage = result_perc
        previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

        if previous_percentage != 0:
            no_progress = False
            time.sleep(reconfig_intervall_seconds)
            break
        else:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] Observed zero progress, waiting...")
        time.sleep(1)

    while(transfer_running):

        # get metrics
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        current_metric_poll_timestamp = datetime.fromisoformat(timestamp)
        current_percentage = result_perc

        throughput_since_last = (FILESIZE * (current_percentage - previous_percentage) / 100) / 1_000_000 / ((current_metric_poll_timestamp - previous_metric_poll_timestamp).total_seconds())
        result_metric = throughput_since_last

        # save all datapoints
        result[metric] = result_metric#[metric]
        result['percentage_progress'] = current_percentage
        result['metric_poll_timestamp'] = current_metric_poll_timestamp
        result['reconfiguration_timestamp'] = reconfiguration_timestamp
        result.update(current_configuration)

        previous_metric_poll_timestamp = current_metric_poll_timestamp
        previous_percentage = current_percentage

        df = pd.DataFrame(result, index=[0])
        df.to_csv(filepath + filename, mode='a', header=(not os.path.isfile(output_file)))
        result = {}

        if current_percentage >= 100:
            transfer_running = False
            break

        #update the cost model with the new result
        print(f"[{datetime.today().strftime('%H:%M:%S')}] Updating Optimizer with result {result_metric}  MB/s, progress at {current_percentage}%")
        optimizer.report(current_configuration, {'uncompressed_throughput':result_metric})

        current_configuration = optimizer.suggest()
        current_parameters = map_configuration(current_configuration)

        # reconfigure trasnfer
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfiguring running transfer with new configuration {current_configuration}")
        reconfiguration_timestamp = datetime.now()
        reconfigure_xdbc_transfer_wrapper(ssh,current_parameters,current_parameters)

        # wait intervcall
        time.sleep(reconfig_intervall_seconds)

    end_transfer_timestmap = datetime.now()

    print(f"Transfer completed in {(end_transfer_timestmap - start_transfer_timestamp).total_seconds()} seconds")
    print(f"Average throughput of  {FILESIZE/1000/1000 / (end_transfer_timestmap - start_transfer_timestamp).total_seconds()} MB/s")


def online_optimization_loop_random(environment, metric, mode, config_space, reconfig_intervall_seconds, ssh):

    config_space_string = get_config_space_string(config_space)
    filename = f"results_random_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    filepath = f"results_online3_{config_space_string}/{environment_to_string(environment)}/random/"
    output_file = filepath + filename
    Path(filepath).mkdir(parents=True, exist_ok=True)

    start_docker_setup(ssh)

    optimizer = Syne_Tune_Ask_Tell(config_space=config_space, metric=metric, mode=mode, underlying='random_search')

    # get inintal configuration
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Querying Optimizer for initial configuration")
    inital_configuration = optimizer.suggest()

    # map config to actual paramter values
    inital_paramters = map_configuration(inital_configuration)

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Starting transfer with initial configuration {inital_configuration}")

    # start transfer
    complete_config = {**inital_paramters, **fixed_parameters}
    start_xdbc_transfer_parameter_wrapper(ssh, environment, complete_config, complete_config )
    reconfiguration_timestamp = datetime.now()
    start_transfer_timestamp = datetime.now()

    # recompute search space to have non-reconfigurable parameters fixed
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reducing search space to keep non reconfigurable parameters fixed")
    reduced_search_space = reduce_search_sapce(inital_configuration)
    search_space_size = calculate_search_space_size(config_space)
    reduced_search_space_size = calculate_search_space_size(reduced_search_space)
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reduced search space has a size of {reduced_search_space_size}, original size was {search_space_size}, reduced to {reduced_search_space_size/search_space_size}")

    # restart optimzer with reduced search space
    optimizer = Syne_Tune_Ask_Tell(config_space=reduced_search_space, metric=metric, mode=mode, underlying='random_search')

    transfer_running = True
    current_configuration = inital_configuration
    result = {}

    (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
    previous_percentage = result_perc
    previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

    # add loop that waits till non zero percentage is returned
    no_progress = True
    if previous_percentage != 0:
        no_progress = False

    while(no_progress):
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        previous_percentage = result_perc
        previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

        if previous_percentage != 0:
            no_progress = False
            time.sleep(reconfig_intervall_seconds)
            break
        else:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] Observed zero progress, waiting...")
        time.sleep(1)

    while(transfer_running):

        # get metrics
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        current_metric_poll_timestamp = datetime.fromisoformat(timestamp)
        current_percentage = result_perc

        throughput_since_last = (FILESIZE * (current_percentage - previous_percentage) / 100) / 1_000_000 / ((current_metric_poll_timestamp - previous_metric_poll_timestamp).total_seconds())
        result_metric = throughput_since_last

        # save all datapoints
        result[metric] = result_metric#[metric]
        result['percentage_progress'] = current_percentage
        result['metric_poll_timestamp'] = current_metric_poll_timestamp
        result['reconfiguration_timestamp'] = reconfiguration_timestamp
        result.update(current_configuration)

        previous_metric_poll_timestamp = current_metric_poll_timestamp
        previous_percentage = current_percentage

        df = pd.DataFrame(result, index=[0])
        df.to_csv(filepath + filename, mode='a', header=(not os.path.isfile(output_file)))
        result = {}

        if current_percentage >= 100:
            transfer_running = False
            break

        #update the cost model with the new result
        print(f"[{datetime.today().strftime('%H:%M:%S')}] Updating Optimizer with result {result_metric}  MB/s, progress at {current_percentage}%")
        #optimizer.report(current_configuration, {'uncompressed_throughput':result_metric})

        current_configuration = optimizer.suggest()
        current_parameters = map_configuration(current_configuration)

        # reconfigure trasnfer
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfiguring running transfer with new configuration {current_configuration}")
        reconfiguration_timestamp = datetime.now()
        reconfigure_xdbc_transfer_wrapper(ssh,current_parameters,current_parameters)

        # wait intervcall
        time.sleep(reconfig_intervall_seconds)

    end_transfer_timestmap = datetime.now()

    print(f"Transfer completed in {(end_transfer_timestmap - start_transfer_timestamp).total_seconds()} seconds")
    print(f"Average throughput of  {FILESIZE/1000/1000 / (end_transfer_timestmap - start_transfer_timestamp).total_seconds()} MB/s")




def online_optimization_loop_rgpe(environment, metric, mode, config_space, reconfig_intervall_seconds, ssh):

    config_space_string = get_config_space_string(config_space)
    filename = f"results_rgpe_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    filepath = f"results_online3_{config_space_string}/{environment_to_string(environment)}/rgpe/"
    output_file = filepath + filename
    Path(filepath).mkdir(parents=True, exist_ok=True)

    start_docker_setup(ssh)

    data, suffix = get_transfer_learning_data_for_environment(environment, use_all_environments=False, data_per_env=400)
    optimizer = OpenBox_Ask_Tell(config_space=config_space, metric=metric, mode=mode, underlying='tlbo_rgpe_prf')
    optimizer.load_transfer_learning_history_per_env_from_dataframe(data=data, training_data_per_env=400)

    # get inintal configuration
    inital_configuration = optimizer.suggest()

    # map config to actual paramter values
    inital_paramters = map_configuration(inital_configuration)

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Starting transfer with initial configuration {inital_configuration}")

    # start transfer
    complete_config = {**inital_paramters, **fixed_parameters}
    start_xdbc_transfer_parameter_wrapper(ssh, environment, complete_config, complete_config )
    reconfiguration_timestamp = datetime.now()
    start_transfer_timestamp = datetime.now()

    # recompute search space to have non-reconfigurable parameters fixed
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reducing search space to keep non reconfigurable parameters fixed")
    reduced_search_space = reduce_search_sapce(inital_configuration)
    search_space_size = calculate_search_space_size(config_space)
    reduced_search_space_size = calculate_search_space_size(reduced_search_space)
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reduced search space has a size of {reduced_search_space_size}, original size was {search_space_size}, reduced to {reduced_search_space_size/search_space_size}")

    # cant restart with reduced search space, because training data would not fit.
    #optimizer = OpenBox_Ask_Tell(config_space=reduced_search_space, metric=metric, mode=mode, underlying='bayesian_open_box')

    transfer_running = True
    current_configuration = inital_configuration
    result = {}

    (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
    previous_percentage = result_perc
    previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)


    # add loop that waits till non zero percentage is returned
    no_progress = True
    if previous_percentage != 0:
        no_progress = False

    while(no_progress):
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        previous_percentage = result_perc
        previous_metric_poll_timestamp = datetime.fromisoformat(timestamp)

        if previous_percentage != 0:
            no_progress = False
            time.sleep(reconfig_intervall_seconds)
            break
        else:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] Observed zero progress, waiting...")
        time.sleep(1)


    while(transfer_running):

        # get metrics
        (result_perc,timestamp) = get_xdbc_metrics_wrapper(ssh)
        current_metric_poll_timestamp = datetime.fromisoformat(timestamp)
        current_percentage = result_perc

        throughput_since_last = (FILESIZE * (current_percentage - previous_percentage) / 100) / 1_000_000 / ((current_metric_poll_timestamp - previous_metric_poll_timestamp).total_seconds())
        result_metric = throughput_since_last

        # save all datapoints
        result[metric] = result_metric#[metric]
        result['percentage_progress'] = current_percentage
        result['metric_poll_timestamp'] = current_metric_poll_timestamp
        result['reconfiguration_timestamp'] = reconfiguration_timestamp
        result.update(current_configuration)

        previous_metric_poll_timestamp = current_metric_poll_timestamp
        previous_percentage = current_percentage

        df = pd.DataFrame(result, index=[0])
        df.to_csv(filepath + filename, mode='a', header=(not os.path.isfile(output_file)))
        result = {}

        if current_percentage >= 100:
            transfer_running = False
            break

        #update the cost model with the new result
        print(f"[{datetime.today().strftime('%H:%M:%S')}] Updating Optimizer with result {result_metric}  MB/s, progress at {current_percentage}%")
        optimizer.report(current_configuration, {'uncompressed_throughput':result_metric})

        # query cost model for next parameter configuration
        current_configuration = optimizer.suggest()
        current_parameters = map_configuration(current_configuration)

        # reconfigure trasnfer
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfiguring running transfer with new configuration {current_configuration}")
        reconfiguration_timestamp = datetime.now()
        reconfigure_xdbc_transfer_wrapper(ssh,current_parameters,current_parameters)

        # wait intervcall
        time.sleep(reconfig_intervall_seconds)

    end_transfer_timestmap = datetime.now()

    print(f"Transfer completed in {(end_transfer_timestmap - start_transfer_timestamp).total_seconds()} seconds")
    print(f"Average throughput of  {FILESIZE/1000/1000 / (end_transfer_timestmap - start_transfer_timestamp).total_seconds()} MB/s")




if __name__ == '__main__':
    main()