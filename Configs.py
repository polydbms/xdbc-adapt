import sys

import numpy as np
from scipy.stats import qmc
from syne_tune.config_space import choice, ordinal, Integer, Float
from openbox import space as sp

DEFAULT_LOGGING = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(name)s] [%(levelname)s] : %(message)s',
            'datefmt': '%H:%M:%S' },
    },
    'handlers': {
        'console':  {'class': 'logging.StreamHandler',
                     'formatter': "standard",
                     'level': 'DEBUG',
                     'stream': sys.stdout}

    },
    'loggers': {
        __name__:   {'level': 'DEBUG',
                     'handlers': ['console'],
                     'propagate': False,
                     'force': True},
    }
}


fixed_parameters = {
    "xdbc_version": 11,
    "run": 1,
    "client_readmode": 1,
    "network_latency": 0,
    "network_loss": 0,
    "table": "lineitem_sf100",
    'src': 'csv',
    'target': 'csv',
    'server_container': 'xdbcserver',
    'client_container': 'xdbcclient',
    'src_format': 1,
    'target_format': 1
}

fixed_parameters_PG_CSV = { #
    "xdbc_version": 11,
    "run": 1,
    "client_readmode": 1,
    "network_latency": 0,
    "network_loss": 0,
    "table": "lineitem_sf10",
    'src': 'postgres',
    'target': 'csv',
    'server_container': 'xdbcserver',
    'client_container': 'xdbcclient',
    'src_format': 1,
    'target_format': 1
}

fixed_parameters_PG_PD = { #
    "xdbc_version": 11,
    "run": 1,
    "client_readmode": 1,
    "network_latency": 0,
    "network_loss": 0,
    "table": "lineitem_sf10",
    'src': 'postgres',
    'target': 'pandas',
    'server_container': 'xdbcserver',
    'client_container': 'xdbcpython',
    'src_format': 1,
    'target_format': 1
}

variable_parameter_baseline = {
    "compression_lib" : 'nocomp',
    "bufpool_size": 16,
    "buffer_size": 256,
    "send_par": 2,
    "write_par": 2,
    "decomp_par": 2,
    "read_par": 2,
    "deser_par": 2,
    "ser_par": 2,
    "comp_par": 2
}

config_space_keys = ['compression',
                     'format',
                     'client_bufpool_factor',
                     'server_bufpool_factor',
                     'buffer_size',
                     'send_par',
                     'write_par',
                     'decomp_par',
                     'read_par',
                     'deser_par',
                     'ser_par',
                     'comp_par']


config_space_variable_parameters_generalized_FOR_NEW_ITERATION_10_5_M = \
    [{'name': "compression", 'type': 'categorical', 'domain': ["nocomp", "zstd", "lz4", "lzo", "snappy"]}, # remove either lzo or lz4, ss size from 10.5M to 8.4M
     {'name': "format", 'type': 'categorical', 'domain': [1, 2]},
     {'name': "client_bufpool_factor", 'type': 'discrete', 'domain': [1, 2, 4, 6]}, # sum(threadcounts) * 2 * factor
     {'name': "server_bufpool_factor", 'type': 'discrete', 'domain': [1, 2, 4, 6]},
     {'name': "buffer_size", 'type': 'discrete', 'domain': [64, 256, 512, 1024]},
     {'name': "send_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     {'name': "write_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     {'name': "decomp_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     {'name': "read_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     {'name': "deser_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     {'name': "ser_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     {'name': "comp_par", 'type': 'discrete', 'domain': [1, 4, 8, 16]},
     ]

config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE = \
    [{'name': "compression", 'type': 'categorical', 'domain': ["nocomp", "zstd", "lz4", "lzo", "snappy"]}, # remove either lzo or lz4, ss size from 10.5M to 8.4M
     {'name': "format", 'type': 'categorical', 'domain': [1, 2]},
     {'name': "client_bufpool_factor", 'type': 'integer', 'lower': 1, 'upper': 8},
     {'name': "server_bufpool_factor", 'type': 'integer', 'lower': 1,  'upper': 8},
     {'name': "buffer_size", 'type': 'integer', 'lower': 64, 'upper': 1024},
     {'name': "send_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "write_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "decomp_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "read_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "deser_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "ser_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "comp_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     ]

config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz = \
    [{'name': "compression", 'type': 'categorical', 'domain': ["nocomp", "zstd", "lz4", "lzo", "snappy"]}, # remove either lzo or lz4, ss size from 10.5M to 8.4M
     {'name': "format", 'type': 'categorical', 'domain': [1, 2]},
     {'name': "client_bufpool_factor", 'type': 'integer', 'lower': 1, 'upper': 8},
     {'name': "server_bufpool_factor", 'type': 'integer', 'lower': 1,  'upper': 8},
     {'name': "buffer_size", 'type': 'discrete', 'domain': [64, 256, 512, 1024]},
     #{'name': "buffer_size", 'type': 'discrete', 'domain': [64, 256]},
     {'name': "send_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     #{'name': "send_par", 'type': 'categorical', 'domain': [1]},
     {'name': "write_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "decomp_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "read_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "deser_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "ser_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     {'name': "comp_par", 'type': 'integer', 'lower': 1, 'upper': 16},
     ]

config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_DISCRETE_EX_BufSiz = \
   [{'name': "compression", 'type': 'categorical', 'domain': ["nocomp", "zstd", "lz4", "lzo", "snappy"]},
    {'name': "format", 'type': 'categorical', 'domain': [1, 2]},
    {'name': "client_bufpool_factor", 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
    {'name': "server_bufpool_factor", 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
    {'name': "buffer_size", 'type': 'discrete', 'domain': [64, 256, 512, 1024]},
    {'name': "send_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    {'name': "write_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    {'name': "decomp_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    {'name': "read_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    {'name': "deser_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    {'name': "ser_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    {'name': "comp_par", 'type': 'discrete', 'domain': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]},
    ]




username_cloud_7 = "bene"
all_hosts_cloud_7 = ["cloud-7.dima.tu-berlin.de",
                     "cloud-8.dima.tu-berlin.de",
                     "cloud-9.dima.tu-berlin.de",
                     "cloud-10.dima.tu-berlin.de"]


username_big_cluster = "bdidrich-ldap"
big_cluster_main_host = "sr630-wn-a-01.dima.tu-berlin.de"
reserved_hosts_big_cluster = [
    #6
    "sr630-wn-a-11",
    "sr630-wn-a-12",
    #"sr630-wn-a-13",
    "sr630-wn-a-14",
    "sr630-wn-a-15",
    "sr630-wn-a-16",
    #12
    "sr630-wn-a-19",
    "sr630-wn-a-20",
    "sr630-wn-a-21",
    "sr630-wn-a-22",
    "sr630-wn-a-23",
    "sr630-wn-a-24",
    "sr630-wn-a-25",
    "sr630-wn-a-26",
    "sr630-wn-a-27",
    "sr630-wn-a-28",
    "sr630-wn-a-29",
    "sr630-wn-a-30",
    # permanent testing nodes 9
    "sr630-wn-a-31",
    "sr630-wn-a-32",
    "sr630-wn-a-33",
    "sr630-wn-a-34",
    "sr630-wn-a-35",
    "sr630-wn-a-36",
    "sr630-wn-a-37",
    "sr630-wn-a-38",
    "sr630-wn-a-39",
    # 11
    "sr630-wn-a-40",
    "sr630-wn-a-41",
    "sr630-wn-a-42",
    "sr630-wn-a-43",
    "sr630-wn-a-44",
    "sr630-wn-a-45",
    "sr630-wn-a-46",
    "sr630-wn-a-47",
    "sr630-wn-a-48",
    #"sr630-wn-a-49",
    #"sr630-wn-a-50",

]



def get_username_for_host(ssh_host):
    if ssh_host in all_hosts_cloud_7:
        return username_cloud_7
    if ssh_host in reserved_hosts_big_cluster or ssh_host == big_cluster_main_host:
        return username_big_cluster
    else:
        return "unknown ssh host"


def create_complete_config(environment, metric, library="syne_tune",config=None):
    """
    Takes a configuration space and adds all other needed parameters for a data transfer.
    The optimization algorithms only know and supply the variable parameters, so all fixed parameters need to be added
    for it to be a complete configuration

    Current supported libraries:
        - our generalized format
        - aws syne-tune
        - openbox
        - gpyopt

    Parameters:
        environment (list[dicts]): The generalized configuration space to convert.
        metric (str): the metric to be optimized.
        library (str): The name of the library to convert the configuration space to.
        config (str): The configuration to be completed.

    Returns:
        : Library-specific complete configuration.
    """

    if library == "dict_add" or library == 'dict':

        config_copy = dict(config)

        for key, value in fixed_parameters.items():
            config_copy[key] = value

        # add environment parameters to the config space
        for key, value in environment.items():
            config_copy[key] = value

        config_copy['metric'] = metric


        #add any missing variable parameters :
        for key, value in variable_parameter_baseline.items():
            #if config_copy[key] is None:
            if key not in config_copy.keys():
                config_copy[key] = value

        return config_copy


def convert_from_general(config_space, library):
    """
    Converts a config_space from a generalized format to the proprietary configuration spaces needed for different libraries.
    Current supported libraries:
        - aws syne-tune
        - openbox

    Parameters:
        config_space (list[dicts]): The generalized configuration space to convert.
        library (str): The name of the library to convert the configuration space to.

    Returns:
        : Library-specific configuration space.
    """
    library_config_space = None

    if library == 'syne_tune':
        library_config_space = {}
        for elem in config_space:
            if elem['type'] == 'categorical':
                library_config_space[elem['name']] = choice(elem['domain'])
            if elem['type'] == 'discrete':
                library_config_space[elem['name']] = ordinal(elem['domain'])
            if elem['type'] == 'integer':
                library_config_space[elem['name']] = Integer(elem['lower'], elem['upper'])
            if elem['type'] == 'real':
                library_config_space[elem['name']] = Float(elem['lower'], elem['upper'])

    if library == 'open_box':
        library_config_space = sp.Space()
        for elem in config_space:
            if elem['type'] == 'categorical':
                library_config_space.add_variable(sp.Categorical(elem['name'], elem['domain']))
            if elem['type'] == 'discrete':
                library_config_space.add_variable(sp.Ordinal(elem['name'], elem['domain']))
            if elem['type'] == 'integer':
                library_config_space.add_variable(sp.Int(elem['name'], elem['lower'], elem['upper']))
            if elem['type'] == 'real':
                library_config_space.add_variable(sp.Real(elem['name'], elem['lower'], elem['upper']))

    return library_config_space


def get_config_space_string(config_space):
    config_space_string = ""

    if config_space == config_space_variable_parameters_generalized_FOR_NEW_ITERATION_10_5_M:
        config_space_string = "10_5M"
    elif config_space == config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE:
        config_space_string = "10_5M_flex"
    elif config_space == config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz:
        config_space_string = "10_5M_semi_flex"
    elif config_space == config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_DISCRETE_EX_BufSiz:
        config_space_string = "10_5M_semi_flex"
    return config_space_string# + "PG_CSV" # todo


def environment_to_string(environment):
    return f"S{environment['server_cpu']}_C{environment['client_cpu']}_N{environment['network']}"


def environment_values_to_string(server_cpu,client_cpu,network):
    return f"S{int(round(server_cpu,0))}_C{int(round(client_cpu,0))}_N{int(round(network,0))}"


def unparse_environment(env_string):
    try:
        parts = env_string.split('_')
        server_cpu = int(parts[0][1:])
        client_cpu = int(parts[1][1:])
        network = int(parts[2][1:])
        return {
            "server_cpu": server_cpu,
            "client_cpu": client_cpu,
            "network": network
        }
    except (IndexError, ValueError):
        raise ValueError(f"Invalid input format. Expected format: 'SXX_CXX_NXXX', got {env_string}")


def unparse_environment_float(env_string):
    try:
        parts = env_string.split('_')
        server_cpu = float(parts[0][1:])
        client_cpu = float(parts[1][1:])
        network = float(parts[2][1:])
        return {
            "server_cpu": server_cpu,
            "client_cpu": client_cpu,
            "network": network
        }
    except (IndexError, ValueError):
        raise ValueError(f"Invalid input format. Expected format: 'SXX_CXX_NXXX', got {env_string}")
