from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

from Configs import \
    config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz, create_complete_config
from environments import *

#from experiments.model_optimizer.main_experiments import get_transfer_learning_data_for_environment

from model_implementations.Weighted_Combination_RF_Cost_Model import \
    Per_Environment_RF_Cost_Model
from model_implementations.lhs_search_optimizer import LHS_Search_Optimizer



def get_first_suggestion(search_space, n_queries, cost_model):
    '''
    Idea : try to find a configuration, for which the individual predictions have the highest variance.
    this way we can more accuratly assign it to a cluster afterwards
    '''
    searcher = LHS_Search_Optimizer(config_space=search_space, n_samples=n_queries)

    best_config = None
    max_spread = -np.inf

    start = datetime.now()

    for i_inner in range(1, n_queries + 1):

        #get a configration
        suggested_config = searcher.suggest()

        #get the cost prediction for that configuration
        prediction_vector = cost_model.get_predictions_vector(data=suggested_config)
        #spread = np.std(prediction_vector)
        diff_matrix = np.abs(prediction_vector[:, None] - prediction_vector[None, :])
        spread = np.sum(diff_matrix) / 2
        if spread > max_spread:
            max_spread = spread
            best_config = suggested_config
    end = datetime.now()
    print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] running {n_queries} surrogate transfers took {(end - start).total_seconds()} seconds")

    return best_config



def get_next_suggestion_2_phase_search(search_space, n_queries, cost_model, environment, mode, metric):
    searcher = LHS_Search_Optimizer(config_space=search_space, n_samples=int(n_queries/2)+1)

    configurations = []

    start = datetime.now()

    for i_inner in range(1, int(n_queries/2) + 1):

        #get a configration
        suggested_config = searcher.suggest()

        #get the cost prediction for that configuration
        result_cost_model = cost_model.predict(data=suggested_config, target_environment=environment, print_wieghts=False)
        result_metric = float(result_cost_model[metric])
        configurations.append({'config': suggested_config, 'performance': result_metric})

    end = datetime.now()
    print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] running {int(n_queries/2)} surrogate transfers took {(end - start).total_seconds()} seconds")


    if mode == 'max':
        best_performance = max(entry['performance'] for entry in configurations)
        threshold = best_performance * 0.1
        top_configurations = [
            entry for entry in configurations
            if entry['performance'] >= best_performance - threshold
        ]
    elif mode == 'min':
        best_performance = min(entry['performance'] for entry in configurations)
        threshold = best_performance * 0.1
        top_configurations = [
            entry for entry in configurations
            if entry['performance'] <= best_performance + threshold
        ]

    sub_search_space = []
    for param_entry in search_space:
        param_name = param_entry['name']
        param_type = param_entry['type']

        observed_values = set()
        for entry in top_configurations:
            config = entry["config"]
            if param_name in config:
                observed_values.add(config[param_name])

        if not observed_values:
            sub_search_space.append(param_entry)
            continue

        if param_type in ["categorical", "discrete"]:
            original_domain = param_entry.get("domain", [])
            new_domain = [val for val in original_domain if val in observed_values]

            new_entry = param_entry.copy()
            new_entry["domain"] = new_domain
            sub_search_space.append(new_entry)

        elif param_type == "integer":
            try:
                numeric_values = [int(v) for v in observed_values]
            except Exception:
                numeric_values = list(observed_values)
            new_lower = min(numeric_values)
            new_upper = max(numeric_values)

            new_entry = param_entry.copy()
            new_entry["lower"] = int(new_lower)
            new_entry["upper"] = int(new_upper)
            sub_search_space.append(new_entry)

        else:
            sub_search_space.append(param_entry)
    '''
    print("Sub search space based on top configurations:")
    for entry in sub_search_space:
        print(entry)

    original_search_space_size = calculate_search_space_size(search_space)
    sub_search_space_size = calculate_search_space_size(sub_search_space)

    print(f"Original Search Space Size: {original_search_space_size}")
    print(f"Sub Search Space Size:      {sub_search_space_size}")
    print(f"Reduced size to {(sub_search_space_size/original_search_space_size)*100:.7f} %")

    print("Now exploring Sub Search Space")
    '''
    return get_next_suggestion(sub_search_space,int(n_queries/2),cost_model,environment,mode,metric)



def calculate_search_space_size(search_space):
    total_size = 1
    for param in search_space:
        p_type = param.get('type')
        if p_type in ('categorical', 'discrete'):
            domain = param.get('domain', [])
            total_size *= len(domain)
        elif p_type == 'integer':
            lower = param.get('lower')
            upper = param.get('upper')
            total_size *= (upper - lower + 1)
    return int(total_size)

def get_next_suggestion(search_space, n_queries, cost_model, environment, mode, metric):

        searcher = LHS_Search_Optimizer(config_space=search_space, n_samples=n_queries)

        configurations = []

        start = datetime.now()

        for i_inner in range(1, n_queries + 1):

            #get a configration
            suggested_config = searcher.suggest()

            #get the cost prediction for that configuration
            result_cost_model = cost_model.predict(data=suggested_config, target_environment=environment, print_wieghts=False)
            result_metric = float(result_cost_model[metric])
            configurations.append({'config': suggested_config, 'performance': result_metric})

        end = datetime.now()

        best_performance = max(entry['performance'] for entry in configurations)

        top_configurations = [
            entry for entry in configurations
            if entry['performance'] == best_performance
        ]

        top_config = top_configurations[0]['config']

        prediction = cost_model.predict(top_config,target_environment=environment, print_wieghts=False)


        print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] running {n_queries} surrogate transfers took {(end - start).total_seconds()} seconds")
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] best found configuration has predicted metric of {prediction}  ({metric})")
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] best predicted config {top_config}")


        return top_config








        # Idea : instead of taking the single best prediction, take the average of all top n best predictions. Might make it more robust.

        #sorted_configurations = sorted(configurations, key=lambda x: x['performance'], reverse=True)
        #top_n = max(1, len(sorted_configurations) // 10)
        #top_configurations = sorted_configurations[:top_n]

        '''

        if mode == 'max':

            best_performance = max(entry['performance'] for entry in configurations)
            threshold = best_performance * 0.025
            top_configurations = [
                entry for entry in configurations
                if entry['performance'] >= best_performance - threshold
            ]
        elif mode == 'min':

            best_performance = min(entry['performance'] for entry in configurations)
            threshold = best_performance * 0.025
            top_configurations = [
                entry for entry in configurations
                if entry['performance'] <= best_performance + threshold
            ]

        discrete_params = {"compression", "format", "buffer_size"}

        param_numeric_sums = defaultdict(float)
        param_numeric_counts = defaultdict(int)
        param_discrete_counts = defaultdict(Counter)

        for entry in top_configurations:
            config = entry["config"]
            for param, value in config.items():
                if param in discrete_params:
                    param_discrete_counts[param][value] += 1
                else:
                    try:
                        numeric_value = float(value)
                        param_numeric_sums[param] += numeric_value
                        param_numeric_counts[param] += 1
                    except Exception:
                        param_discrete_counts[param][value] += 1


        averaged_config = {}

        for param in param_numeric_sums:
            averaged_config[param] = int(round(param_numeric_sums[param] / param_numeric_counts[param]))

        for param, counter in param_discrete_counts.items():
            most_common_val = counter.most_common(1)[0][0]
            averaged_config[param] = most_common_val


        prediction = cost_model.predict(averaged_config,target_environment=environment, print_wieghts=False)




        print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] running {n_queries} surrogate transfers took {(end - start).total_seconds()} seconds")
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] best found configuration has predicted metric of {prediction}  ({metric})")
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] best predicted config {averaged_config}")

        return averaged_config
        '''


def get_next_suggestion_2_phase_search_batch(search_space, n_queries, cost_model, environment, mode, metric):
    searcher = LHS_Search_Optimizer(config_space=search_space, n_samples=int(n_queries/2))

    configurations = searcher.get_all_configurations()

    start = datetime.now()

    prediction_results = cost_model.predict_batch_01(
        data_list=configurations,
        target_environment=environment
    )

    end = datetime.now()
    #print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] running {int(n_queries/2)} surrogate transfers took {(end - start).total_seconds()} seconds")


    if mode == 'max':
        best_performance = max(result[1] for result in prediction_results)
        threshold = abs(best_performance * 0.1)
        top_results = [
            result for result in prediction_results
            if result[1] >= best_performance - threshold
        ]
    elif mode == 'min':
        best_performance = min(result[1] for result in prediction_results)
        threshold = abs(best_performance * 0.1) # Using abs for robustness
        top_results = [
            result for result in prediction_results
            if result[1] <= best_performance + threshold
        ]


    sub_search_space = []
    for param_entry in search_space:
        param_name = param_entry['name']
        param_type = param_entry['type']

        observed_values = set()
        for result_tuple in top_results:
            config = result_tuple[0]
            if param_name in config:
                observed_values.add(config[param_name])

        if not observed_values:
            sub_search_space.append(param_entry)
            continue

        if param_type in ["categorical", "discrete"]:
            original_domain = param_entry.get("domain", [])
            new_domain = [val for val in original_domain if val in observed_values]

            new_entry = param_entry.copy()
            new_entry["domain"] = new_domain
            sub_search_space.append(new_entry)

        elif param_type == "integer":
            try:
                numeric_values = [int(v) for v in observed_values]
            except Exception:
                numeric_values = list(observed_values)
            new_lower = min(numeric_values)
            new_upper = max(numeric_values)

            new_entry = param_entry.copy()
            new_entry["lower"] = int(new_lower)
            new_entry["upper"] = int(new_upper)
            sub_search_space.append(new_entry)

        else:
            sub_search_space.append(param_entry)
    '''
    print("Sub search space based on top configurations:")
    for entry in sub_search_space:
        print(entry)

    original_search_space_size = calculate_search_space_size(search_space)
    sub_search_space_size = calculate_search_space_size(sub_search_space)

    print(f"Original Search Space Size: {original_search_space_size}")
    print(f"Sub Search Space Size:      {sub_search_space_size}")
    print(f"Reduced size to {(sub_search_space_size/original_search_space_size)*100:.7f} %")

    print("Now exploring Sub Search Space")
    '''
    return get_next_suggestion_batch(sub_search_space,int(n_queries/2),cost_model,environment,mode,metric)


def get_next_suggestion_batch(search_space, n_queries, cost_model, environment, mode, metric):

    searcher = LHS_Search_Optimizer(config_space=search_space, n_samples=n_queries)

    configurations = searcher.get_all_configurations()

    start = datetime.now()

    prediction_results = cost_model.predict_batch_01(
        data_list=configurations,
        target_environment=environment
    )
    end = datetime.now()

    best_performance = max(result[1] for result in prediction_results)


    top_results = [
        result for result in prediction_results
        if result[1] == best_performance
    ]

    top_config = top_results[0][0]

    prediction = cost_model.predict(top_config,target_environment=environment, print_wieghts=False)


    #print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] running {n_queries} surrogate transfers took {(end - start).total_seconds()} seconds")
    #print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] best found configuration has predicted metric of {prediction}  ({metric})")
    #print(f"[{datetime.today().strftime('%H:%M:%S')}] [Cost_Model_Config_Finder] best predicted config {top_config}")


    return top_config


if __name__ == "__main__":

    search_space = config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz


    mode = 'max'

    metric = 'uncompressed_throughput'
    #metric = 'time'

    #environment = env_S2_C2_N50
    #environment = env_S2_C8_N50
    #environment = env_S4_C16_N50

    #environment = env_S8_C8_N150
    #environment = env_S8_C2_N150
    #environment = env_S8_C16_N150

    #environment = env_S16_C4_N1000
    #environment = env_S16_C8_N1000
    environment = env_S16_C16_N1000


    input_fields = [
        #"client_cpu",
        #"server_cpu",
        #"network",         # if i split the rf by env anyway, why would i need to include env variabels ??
        #"network_latency",
        #"network_loss",
        #"source_system",   # potentialy relevant if variety in training data, also the format
        #"target_system",
        #"table",
        "client_bufpool_factor",
        "server_bufpool_factor",
        "buffer_size",
        "compression",
        "send_par",
        #"rcv_par",
        "write_par",
        "decomp_par",
        "read_par",
        "deser_par",
        "ser_par",
        "comp_par"
    ]

    cost_model = Per_Environment_RF_Cost_Model(input_fields=input_fields,
                                               metric=metric,
                                               data_per_env=500,
                                               underlying="test_cost_model",
                                               cluster=True,
                                               regression_model='xgb',
                                               network_transformation=True,
                                               history_ratio=0.8
                                               )

    data, suffix = get_transfer_learning_data_for_environment(environment, False,400)

    x = data
    y = data[metric]

    cost_model.train(x, y)

    n_queries = 1000


    #config = get_next_suggestion_2_phase_search(search_space, n_queries, cost_model, environment, mode, metric)
    config = get_next_suggestion_2_phase_search_batch(search_space, n_queries, cost_model, environment, mode, metric)


    #config = get_next_suggestion(search_space, n_queries, cost_model, environment, mode, metric)
    #config = get_next_suggestion_batch(search_space, n_queries, cost_model, environment, mode, metric)


