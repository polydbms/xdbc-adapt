from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from syne_tune.optimizer.schedulers import HyperbandScheduler, FIFOScheduler
from syne_tune.optimizer.baselines import BayesianOptimization, RandomSearch, GridSearch, ZeroShotTransfer
from typing import Dict
from datetime import datetime

from syne_tune.backend.trial_status import Trial, Status, TrialResult
from syne_tune.optimizer.scheduler import TrialScheduler
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.schedulers.transfer_learning.quantile_based.quantile_based_searcher import \
    QuantileBasedSurrogateSearcher

import Transfer_Data_Processor
import Metrics
from Configs import *


class Syne_Tune_Ask_Tell():
    available_optimization_algorithms = ["bayesian_syne_tune",
                                         "random_search",
                                         "hyperband",
                                         "asha",
                                         "grid_search"]
    available_transfer_algorithms = ["zero_shot",
                                     "quantile"]

    available_algorithms = available_optimization_algorithms + available_transfer_algorithms

    def __init__(self, config_space, metric='time', mode='min', underlying='bayesian'):
        """
       Initialize the SyneTune Ask-Tell Class.

       Parameters:
           config_space (dict): Configuration space for the optimizer.
           metric (str): Metric to optimize.
           mode (str): Mode of optimization.
           underlying (str): Underlying optimization algorithm.
           history (list): Optional history for transfer learning.
       """
        self.current_suggestions = {}
        self.config_space = convert_from_general(config_space, 'syne_tune')
        self.metric = metric
        self.mode = mode
        self.underlying = underlying + "_syne_tune"#_MEASURING_COMPUTE"

        self.transfer_history = None
        self.scheduler = None
        self.current_suggestions = {}

        self.reset()

    def reset(self):
        """
        Resets the optimizer to remove any previous knowledge of completed evaluations.
        To do this it simply creates the underlying optimizer new.
        """

        if 'bayesian' in self.underlying:
            self.scheduler = AskTellScheduler(
                base_scheduler=BayesianOptimization(config_space=self.config_space, metric=self.metric, mode=self.mode)
            )
        elif "random_search" in self.underlying:
            self.scheduler = AskTellScheduler(
                base_scheduler=RandomSearch(config_space=self.config_space, metric=self.metric, mode=self.mode)
            )
        elif "grid_search" in self.underlying:
            self.scheduler = AskTellScheduler(
                base_scheduler=GridSearch(config_space=self.config_space, metric=self.metric, mode=self.mode)
            )
        elif "hyperband" in self.underlying:
            self.scheduler = AskTellScheduler(
                base_scheduler=HyperbandScheduler(
                    config_space=self.config_space,
                    metric=self.metric,
                    mode=self.mode,
                    resource_attr="epoch",
                    max_resource_attr="epochs",
                    search_options={"debug_log": False},
                    grace_period=2,
                    max_t=1000
                )
            )
        elif "asha" in self.underlying:
            self.scheduler = AskTellScheduler(
                base_scheduler=HyperbandScheduler(
                    config_space=self.config_space,
                    metric=self.metric,
                    mode=self.mode,
                    resource_attr="epoch",
                    max_t=1000,
                )
            )
        elif "zero_shot" in self.underlying:
            if self.transfer_history is not None:
                self.scheduler = AskTellScheduler(
                    base_scheduler=ZeroShotTransfer(
                        config_space=self.config_space,
                        metric=self.metric,
                        mode=self.mode,
                        transfer_learning_evaluations=self.transfer_history,
                        use_surrogates=True
                    )
                )
        elif "quantile" in self.underlying:
            if self.transfer_history is not None:
                self.scheduler = AskTellScheduler(
                    base_scheduler=FIFOScheduler(
                        searcher=QuantileBasedSurrogateSearcher(
                            config_space=self.config_space,
                            metric=self.metric,
                            mode=self.mode,
                            transfer_learning_evaluations=self.transfer_history,
                            use_surrogates=True
                        ),
                        config_space=self.config_space,
                        metric=self.metric,
                        mode=self.mode,
                    )
                )
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
        suggested_trial = self.scheduler.ask()
        self.current_suggestions[frozenset(suggested_trial.config.items())] = suggested_trial
        return suggested_trial.config

    def report(self, config, result):
        """
        Reports the result of a completed trial to the underlying optimization algorithm.

        Parameters:
            config (dict): The configuration that was evaluated.
            result (dict): The result of that configuration.

        """
        if self.metric in result.keys():
            metric_value = result[self.metric]
        else:
            metric_value = {self.metric: Metrics.get_metric(self.metric, result)}
        trial = self.current_suggestions[frozenset(config.items())]
        try:
            self.scheduler.tell(trial, metric_value)
        except:
            print("oh no")

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
        transfer_learning_evaluations = {}

        #for env_key, df in grouped_data.items():
        for env_key, df in cluster_dataframes.items():
            df = df[df['time'].notna() & (df['transfer_id'] > 0)]

            if training_data_per_env != -1:
                #df = df.head(training_data_per_env)
                if f"_n_{training_data_per_env}" not in self.underlying:
                    self.underlying += f"_n_{training_data_per_env}"

            transfer_learning_evaluations[env_key] = TransferLearningTaskEvaluations(
                configuration_space=self.config_space,
                hyperparameters=df[self.config_space.keys()],
                objectives_names=[self.metric],
                objectives_evaluations=np.array(df[self.metric], ndmin=4).T
            )
            print(
                f"[{datetime.today().strftime('%H:%M:%S')}] Loaded History for Environment {env_key} with length {len(df)}")

        self.transfer_history = transfer_learning_evaluations
        self.reset()


"""
copy of ask_tell_scheduler.py from the syne_tune git repo
"""


class AskTellScheduler:
    base_scheduler: TrialScheduler
    trial_counter: int
    completed_experiments: Dict[int, TrialResult]

    def __init__(self, base_scheduler: TrialScheduler):
        """
        Simple interface to use SyneTune schedulers in a custom loop, for example:

        .. code-block:: python

            scheduler = AskTellScheduler(base_scheduler=RandomSearch(config_space, metric=metric, mode=mode))
            for iter in range(max_iterations):
                trial_suggestion = scheduler.ask()
                test_result = target_function(**trial_suggestion.config)
                scheduler.tell(trial_suggestion, {metric: test_result})

        :param base_scheduler: Scheduler to be wrapped
        """
        self.base_scheduler = base_scheduler
        self.trial_counter = 0
        self.completed_experiments = {}

    def ask(self) -> Trial:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        trial_suggestion = self.base_scheduler.suggest(self.trial_counter)
        trial = Trial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.now(),
        )
        self.trial_counter += 1
        return trial

    def tell(self, trial: Trial, experiment_result: Dict[str, float]):
        """
        Feed experiment results back to the Scheduler

        :param trial: Trial that was run
        :param experiment_result: {metric: value} dictionary with experiment results
        """
        trial_result = trial.add_results(
            metrics=experiment_result,
            status=Status.completed,
            training_end_time=datetime.now(),
        )
        self.base_scheduler.on_trial_complete(trial=trial, result=experiment_result)
        self.completed_experiments[trial_result.trial_id] = trial_result

    def best_trial(self, metric: str) -> TrialResult:
        """
        Return the best trial according to the provided metric.

        :param metric: Metric to use for comparison
        """
        if self.base_scheduler.mode == "max":
            sign = 1.0
        else:
            sign = -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metric],
        )
