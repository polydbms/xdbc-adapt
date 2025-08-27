from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from openbox import Advisor, Observation, History

from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import Metrics
from Configs import *
import Transfer_Data_Processor


class OpenBox_Ask_Tell():

    available_optimization_algorithms = ['bayesian_open_box', 'bayesian_gp_open_box', 'bayesian_prf_open_box',]

    available_transfer_algorithms = ['tlbo_rgpe_prf', 'tlbo_sgpr_prf', 'tlbo_topov3_prf',
                                     'tlbo_rgpe_gp', 'tlbo_sgpr_gp', 'tlbo_topov3_gp']



    available_algorithms = available_optimization_algorithms + available_transfer_algorithms


    def __init__(self, config_space, metric='time', mode='time', underlying='bayesian'):
        """
        Initialize the OpenBox Ask-Tell Class.

        Parameters:
            config_space (dict): Configuration space for the optimizer.
            metric (str): Metric to optimize.
            mode (str): Mode of optimization.
            underlying (str): Underlying optimization algorithm.
        """
        self.config_space = convert_from_general(config_space, 'open_box')
        self.metric = metric
        self.mode = mode
        if "open_box" not in underlying:
            self.underlying = f"{underlying}_open_box"#'_clust-03"
        else:
            self.underlying = underlying# + "_clust-03"

        self.history = None
        self.advisor = None
        self.current_suggestions = {}

        self.reset()

    def reset(self):
        """
        Resets the optimizer to remove any previous knowledge of completed evaluations.
        Initializes the appropriate advisor based on the underlying algorithm.
        """
        print(f"[{datetime.today().strftime('%H:%M:%S')}] Resetting the optimizer...")
        bayesian_config = {
            "config_space": self.config_space,
            "acq_type": "ei",
            "surrogate_type": "prf",
            "rand_prob": 0
        }

        transfer_config = {
            "config_space": self.config_space,
            "num_objectives": 1,
            "num_constraints": 0,
            "initial_trials": 0,
            "rand_prob": 0,
            "transfer_learning_history": self.history,
            "acq_type": "ei"
        }

        transfer_algorithms = ['tlbo_rgpe_prf', 'tlbo_sgpr_prf', 'tlbo_topov3_prf', 'tlbo_rgpe_gp', 'tlbo_sgpr_gp',
                               'tlbo_topov3_gp']

        if 'bayesian' in self.underlying:
            if 'bayesian_gp' in self.underlying:
                bayesian_config["surrogate_type"] = "gp"
            elif 'bayesian_prf' in self.underlying:
                bayesian_config["surrogate_type"] = "prf"

            print(f"[{datetime.today().strftime('%H:%M:%S')}] Initializing Bayesian optimization with {bayesian_config['surrogate_type']} surrogates")
            self.advisor = Advisor(**bayesian_config)
        elif any(key in self.underlying for key in transfer_algorithms):
            if not self.history:
                print(f"[{datetime.today().strftime('%H:%M:%S')}] Transfer learning history is empty. Skipping initialization of the transfer learning advisor for underlying algorithm {self.underlying}.")
            else:
                if 'tlbo_rgpe_prf' in self.underlying:
                    transfer_config["surrogate_type"] = "tlbo_rgpe_prf"
                elif 'tlbo_sgpr_prf' in self.underlying:
                    transfer_config["surrogate_type"] = "tlbo_sgpr_prf"
                elif 'tlbo_topov3_prf' in self.underlying:
                    transfer_config["surrogate_type"] = "tlbo_topov3_prf"
                elif 'tlbo_rgpe_gp' in self.underlying:
                    transfer_config["surrogate_type"] = "tlbo_rgpe_gp"
                elif 'tlbo_sgpr_gp' in self.underlying:
                    transfer_config["surrogate_type"] = "tlbo_sgpr_gp"
                elif 'tlbo_topov3_gp' in self.underlying:
                    transfer_config["surrogate_type"] = "tlbo_topov3_gp"
                print(
                    f"[{datetime.today().strftime('%H:%M:%S')}] Initializing transfer learning algorithm: {self.underlying}")
                self.advisor = Advisor(**transfer_config)
        else:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] Unknown underlying algorithm: {self.underlying}")
            raise ValueError(f"Unknown underlying algorithm: {self.underlying}")

        self.current_suggestions = {}

    def suggest(self):
        """
        Gets the next suggested trial from the underlying optimization algorithm and returns it as a dictionary.

        Returns:
            dict: The next suggested configurations.
        """
        suggested_config = self.advisor.get_suggestion()
        self.current_suggestions[frozenset(suggested_config.items())] = suggested_config
        return suggested_config.get_dictionary()

    def report(self, config, result):
        """
        Reports the result of a completed trial to the underlying optimization algorithm.

        Parameters:
            config (dict): The configuration that was evaluated.
            result (dict): The result of that configuration.

        """
        #complete_config = self.current_suggestions[frozenset(config.items())]
        if self.metric in result.keys():
            metric_value = result[self.metric]
        else:
            metric_value = Metrics.get_metric(self.metric, result)

        # openbox always minimizes the objective
        if self.mode == 'max':
            metric_value = metric_value * -1
        #observation = Observation(config=complete_config, objectives=metric_value)

        config = {k: config[k] for k in config_space_keys if k in config}#

        if config['compression'] == "no_comp":
            config['compression'] = "nocomp"

        observation = Observation(config=Configuration(self.config_space, config), objectives=metric_value)
        self.advisor.update_observation(observation)


    def load_transfer_learning_history_per_env_from_dataframe(self, data, training_data_per_env=-1):
        """
        Loads a list of files as previous evaluations into the underlying optimizer, creating one history per environment.
        Groups evaluations by environment and loads them as a transfer learning history into the algorithm.

        Parameters:
            file_list (list): List of file paths to load.
            training_data_per_env (int): Number of samples to use per environment (-1 for no limit).
        """
        cluster_dataframes = Transfer_Data_Processor.process_data(data,training_data_per_env)


        self.underlying = self.underlying + "_with_clustering"

        # Create transfer learning histories
        transfer_learning_history = []

        #for env_key, df in grouped_data.items():
        for env_key, df in cluster_dataframes.items():
            df = df[df['time'].notna() & (df['transfer_id'] > 0)]

            if training_data_per_env != -1:
                #df = df.head(training_data_per_env)
                if f"_n_{training_data_per_env}" not in self.underlying:
                    self.underlying += f"_n_{training_data_per_env}"

            history = self._create_history(env_key, df)
            transfer_learning_history.append(history)
            print(
                f"[{datetime.today().strftime('%H:%M:%S')}] Loaded History for Environment {env_key} with length {len(df)}")

        self.history = transfer_learning_history
        self.reset()

    def _create_history(self, env_key, df):
        """
        Creates a history object for a given environment.

        Parameters:
            env_key (tuple): The environment key (e.g., combination of server, client, and network).
            df (DataFrame): DataFrame containing the configuration and metrics for the environment.

        Returns:
            History: A populated History object.
        """
        history = History(task_id=f'history_{env_key}', config_space=self.config_space)

        for _, row in df.iterrows():
            config = {
                'compression': row['compression'],
                'format': row['format'],
                'client_bufpool_factor': int(row['client_bufpool_factor']),
                'server_bufpool_factor': int(row['server_bufpool_factor']),
                'buffer_size': int(row['buffer_size']),
                'send_par': int(row['send_par']),
                'write_par': int(row['write_par']),
                'decomp_par': int(row['decomp_par']),
                'read_par': int(row['read_par']),
                'deser_par': int(row['deser_par']),
                'ser_par': int(row['ser_par']),
                'comp_par': int(row['comp_par'])
            }

            metric = row[self.metric]

            if self.mode == 'max':
                metric = metric * -1

            observation = Observation(config=Configuration(self.config_space, config), objectives=metric)
            history.update_observation(observation)

        return history
