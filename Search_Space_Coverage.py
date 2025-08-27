import itertools
import numpy as np
from scipy.stats import qmc
import math
from scipy.stats import kstest

from model_implementations.syne_tune_ask_tell import Syne_Tune_Ask_Tell

search_space = [
    {'name': "compression", 'type': 'categorical', 'domain': ["nocomp", "zstd", "lz4", "lzo", "snappy"]}, # remove either lzo or lz4, ss size from 10.5M to 8.4M
    {'name': "format", 'type': 'categorical', 'domain': [1, 2]},
    {'name': "client_bufpool_factor", 'type': 'integer', 'lower': 1, 'upper': 8},
    {'name': "server_bufpool_factor", 'type': 'integer', 'lower': 1,  'upper': 8},
    {'name': "buffer_size", 'type': 'discrete', 'domain': [64, 256, 512, 1024]},
    {'name': "send_par", 'type': 'integer', 'lower': 1, 'upper': 16},
    {'name': "write_par", 'type': 'integer', 'lower': 1, 'upper': 16},
    {'name': "decomp_par", 'type': 'integer', 'lower': 1, 'upper': 16},
    {'name': "read_par", 'type': 'integer', 'lower': 1, 'upper': 16},
    {'name': "deser_par", 'type': 'integer', 'lower': 1, 'upper': 16},
    {'name': "ser_par", 'type': 'integer', 'lower': 1, 'upper': 16},
    {'name': "comp_par", 'type': 'integer', 'lower': 1, 'upper': 16},
]

max_distance = 3.464101


def config_custom_distance(config1, config2):
    squared_sum = 0.0

    if config1['compression'] != config2['compression']:
        squared_sum += 1
    if config1['format'] != config2['format']:
        squared_sum += 1

    norm1 = normalize(config1['buffer_size'], 64, 1024)
    norm2 = normalize(config2['buffer_size'], 64, 1024)
    squared_sum += (norm1 - norm2) ** 2

    norm1 = normalize(config1['client_bufpool_factor'], 1, 8)
    norm2 = normalize(config2['client_bufpool_factor'], 1, 8)
    squared_sum += (norm1 - norm2) ** 2

    norm1 = normalize(config1['server_bufpool_factor'], 1, 8)
    norm2 = normalize(config2['server_bufpool_factor'], 1, 8)
    squared_sum += (norm1 - norm2) ** 2

    parallel_keys = [
        'read_par',
        'deser_par',
        'comp_par',
        'send_par',
        'decomp_par',
        'ser_par',
        'write_par'
    ]
    for key in parallel_keys:
        norm1 = normalize(config1[key], 1, 16)
        norm2 = normalize(config2[key], 1, 16)
        squared_sum += (norm1 - norm2) ** 2

    return math.sqrt(squared_sum)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def pairwise_distance_for_each(n=100):
    configurations_lhs = generate_lhs_configs_flex(search_space,n)
    configurations_random = generate_random_configs(search_space,n)
    configurations_grid = generate_grid_configs(search_space,n)

    print("Results for calculating the Average Pairwise Distances. Higher is better.")


    total_distance_lhs = 0.0
    count_lhs = 0
    for config1, config2 in itertools.combinations(configurations_lhs, 2):
        dist = config_custom_distance(config1, config2)
        norm_dist = dist / max_distance
        total_distance_lhs += norm_dist
        count_lhs += 1
    average_pairwise_distance_lhs = total_distance_lhs / count_lhs
    print(f"Average normalized pairwise distance LHS: {average_pairwise_distance_lhs}")



    total_distance_rnd = 0.0
    count_rnd = 0
    for config1, config2 in itertools.combinations(configurations_random, 2):
        dist = config_custom_distance(config1, config2)
        norm_dist = dist / max_distance
        total_distance_rnd += norm_dist
        count_rnd += 1
    average_pairwise_distance_rnd = total_distance_rnd / count_rnd
    print(f"Average normalized pairwise distance RND: {average_pairwise_distance_rnd}")


    total_distance_grd = 0.0
    count_grd = 0
    for config1, config2 in itertools.combinations(configurations_grid, 2):
        dist = config_custom_distance(config1, config2)
        norm_dist = dist / max_distance
        total_distance_grd += norm_dist
        count_grd += 1
    average_pairwise_distance_grd = total_distance_grd / count_grd
    print(f"Average normalized pairwise distance GRD: {average_pairwise_distance_grd}")





compression_library_order = {
    "nocomp": 0,
    "zstd": 1,
    "lz4": 2,
    "lzo": 3,
    "snappy": 4
}
intermediate_format_order = {
    1: 0,
    2: 1
}

def config_to_point(config):
    """
    Convert a configuration dictionary to a point in [0,1]^d.
    """
    point = []

    comp_val = compression_library_order[config['compression']]
    comp_norm = comp_val / (len(compression_library_order) - 1)
    point.append(comp_norm)

    intf_val = intermediate_format_order[config['format']]
    intf_norm = intf_val / (len(intermediate_format_order) - 1)
    point.append(intf_norm)

    buf_norm = (config['buffer_size'] - 64) / (1024 - 64)
    point.append(buf_norm)

    client_bp_norm = (config['client_bufpool_factor'] - 1) / (8 - 1)
    point.append(client_bp_norm)

    server_bp_norm = (config['server_bufpool_factor'] - 1) / (8 - 1)
    point.append(server_bp_norm)

    parallel_keys = [
        'read_par',
        'deser_par',
        'comp_par',
        'send_par',
        'decomp_par',
        'ser_par',
        'write_par'
    ]
    for key in parallel_keys:
        par_norm = (config[key] - 1) / (16 - 1)
        point.append(par_norm)

    return np.array(point)

def average_nearest_neighbor_distance(points, n_candidate=10000):
    points = np.array(points)
    n_points, d = points.shape
    candidate_points = np.random.rand(n_candidate, d)

    total_distance = 0.0
    for candidate in candidate_points:
        distances = np.linalg.norm(points - candidate, axis=1)
        total_distance += np.min(distances)

    return total_distance / n_candidate

def compute_average_nn_for_configurations(configurations, n_candidate=10000):
    points = [config_to_point(cfg) for cfg in configurations]
    return average_nearest_neighbor_distance(points, n_candidate=n_candidate)


def average_nearest_nn_for_each(n=100):
    configurations_lhs = generate_lhs_configs_flex(search_space,n)
    configurations_random = generate_random_configs(search_space,n)
    configurations_grid = generate_grid_configs(search_space,n)

    print("Results for calculating the Average Nearest Neighbor Distance. Higher is better.")


    avg_nn_dist_lhs = compute_average_nn_for_configurations(configurations_lhs)
    print(f"Average nearest neigbour distance LHS: {avg_nn_dist_lhs}")

    avg_nn_dist_rnd = compute_average_nn_for_configurations(configurations_random)
    print(f"Average nearest neigbour distance RND: {avg_nn_dist_rnd}")

    avg_nn_dist_grd = compute_average_nn_for_configurations(configurations_grid)
    print(f"Average nearest neigbour distance GRD: {avg_nn_dist_grd}")


def maximin_distance(points):
    min_dist = float('inf')
    for p1, p2 in itertools.combinations(points, 2):
        d = np.linalg.norm(p1 - p2)#(p1, p2)
        if d < min_dist:
            min_dist = d
    return min_dist


def compute_maximin_distance_for_configurations(configurations):
    points = [config_to_point(cfg) for cfg in configurations]
    return maximin_distance(points) / max_distance


def maximin_distance_for_each(n=100):
    configurations_lhs = generate_lhs_configs_flex(search_space,n)
    configurations_random = generate_random_configs(search_space,n)
    configurations_grid = generate_grid_configs(search_space,n)

    print("Results for calculating the Maximin Distance. Higher is better.")


    maximin_lhs = compute_maximin_distance_for_configurations(configurations_lhs)
    print(f"Maximin distance LHS: {maximin_lhs}")

    maximin_random = compute_maximin_distance_for_configurations(configurations_random)
    print(f"Maximin distance RND: {maximin_random}")

    maximin_grid = compute_maximin_distance_for_configurations(configurations_grid)
    print(f"Maximin distance GRD: {maximin_grid}")


def compute_ks_statistics_for_configurations(configurations):
    points = np.array([config_to_point(cfg) for cfg in configurations])
    n_points, d = points.shape

    ks_results = []
    for i in range(d):
        dimension_data = points[:, i]
        stat, p_value = kstest(dimension_data, 'uniform', args=(0, 1))
        ks_results.append((i, stat, p_value))

    return ks_results

def aggregate_ks_statistics(ks_results, method='average'):
    stats = [stat for _, stat, _ in ks_results]
    if method == 'average':
        return sum(stats) / len(stats)
    elif method == 'max':
        return max(stats)


def ks_for_each(n=100):
    configurations_lhs = generate_lhs_configs_flex(search_space,n)
    configurations_random = generate_random_configs(search_space,n)
    configurations_grid = generate_grid_configs(search_space,n)

    print("Results for calculating the Kolmogorov–Smirnov Statistic. Lower is better.")

    ks_lhs = compute_ks_statistics_for_configurations(configurations_lhs)
    avg_ks_lhs = aggregate_ks_statistics(ks_lhs)
    print(f"Avg KS Statistic LHS: {avg_ks_lhs}")

    ks_random = compute_ks_statistics_for_configurations(configurations_random)
    avg_ks_random = aggregate_ks_statistics(ks_random)
    print(f"Avg KS Statistic RND: {avg_ks_random}")

    ks_grid = compute_ks_statistics_for_configurations(configurations_grid)
    avg_ks_grid = aggregate_ks_statistics(ks_grid)
    print(f"Avg KS Statistic GRD: {avg_ks_grid}")



    max_ks_lhs = aggregate_ks_statistics(ks_lhs,'max')
    print(f"Max KS Statistic LHS: {max_ks_lhs}")

    max_ks_random = aggregate_ks_statistics(ks_random,'max')
    print(f"Max KS Statistic RND: {max_ks_random}")

    max_ks_grid = aggregate_ks_statistics(ks_grid,'max')
    print(f"Max KS Statistic GRD: {max_ks_grid}")



def generate_lhs_configs_flex(search_space,n_samples):
    dim = len(search_space)
    sampler = qmc.LatinHypercube(d=dim)
    samples = sampler.random(n=n_samples)

    configs = []
    for sample in samples:
        config = {}
        for i, param in enumerate(search_space):
            p_type = param['type']
            value = None
            if p_type in ['categorical','discrete']:
                domain = param['domain']
                n_values = len(domain)
                idx = int(np.floor(sample[i] * n_values))
                if idx >= n_values:
                    idx = n_values - 1
                value = domain[idx]
            elif p_type == 'integer':
                lower = param['lower']
                upper = param['upper']
                range_size = upper - lower + 1
                idx = int(np.floor(sample[i] * range_size))
                if idx >= range_size:
                    idx = range_size - 1
                value = lower + idx
            else:
                raise ValueError(f"Unknown parameter type {p_type} for parameter {param['name']}.")
            config[param['name']] = value
        configs.append(config)
    return configs

def generate_random_configs(search_space,n_samples):
    configurations = []
    optimizer = Syne_Tune_Ask_Tell(config_space=search_space, underlying='random_search')
    for i in range(0,n_samples):

        config = optimizer.suggest()

        configurations.append(config)

    return configurations

def generate_grid_configs(search_space,n_samples):
    configurations = []
    optimizer = Syne_Tune_Ask_Tell(config_space=search_space, underlying='grid_search')
    for i in range(0,n_samples):

        config = optimizer.suggest()

        configurations.append(config)

    return configurations



if __name__ == "__main__":



    n = 100

    print(f"Metrics to compare search space coverage on n={n} points")

    pairwise_distance_for_each(n)
    print("---------------------------------------------------------------")
    #average_nearest_nn_for_each(n)
    #print("---------------------------------------------------------------")
    maximin_distance_for_each(n)
    print("---------------------------------------------------------------")
    ks_for_each(n)


