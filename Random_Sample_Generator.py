import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import os
import threading
from queue import Queue

from ssh_handler import SSHConnection
import data_transfer_wrapper
from Configs import *
from NestedSSHHandler import NestedSSHClient
from environments import *
from model_implementations.lhs_search_optimizer import LHS_Search_Optimizer
from model_implementations.syne_tune_ask_tell import Syne_Tune_Ask_Tell



def process_configuration(queue, ssh_host):
    """
    #todo

    Parameters:
        queue
        environment (dict): Dictionary containing the values for the environment (server_cpu,client_cpu,network).
        ssh_host (str): The shh host to use for executing the transfers.
        output_file (str): The file into which the results should be saved.
        lock (lock): Lock to lock the output file to not permit parallel writes.
    """
    while not queue.empty():
        config_id, config, environment, output_file, lock = queue.get()
        try:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh_host}] starting config {config_id} for environment {environment_to_string(environment)} {config}")

            complete_config = create_complete_config(environment=environment, metric='time', library='dict', config=config)

            #run transfer
            if ssh_host in reserved_hosts_big_cluster:
                ssh = NestedSSHClient(jump_host=big_cluster_main_host,
                                      jump_username=get_username_for_host(big_cluster_main_host) ,
                                      target_host=ssh_host,
                                      target_username=get_username_for_host(ssh_host))
            else:
                ssh = SSHConnection(ssh_host, get_username_for_host(ssh_host))

            result = data_transfer_wrapper.transfer(config=complete_config, max_retries=1, ssh=ssh, i=config['config_id'])
            ssh.close()

            if result['transfer_id'] == -1:
                result['time'] = -1



            result['client_bufpool_factor'] = complete_config['client_bufpool_factor']
            result['server_bufpool_factor'] = complete_config['server_bufpool_factor']
            result['config_id'] = config_id

            with lock:
                df = pd.DataFrame(result, index=[0])
                if os.path.isfile(output_file):
                    df.to_csv(output_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(output_file, mode='a', header=True, index=False)

            print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh_host}] finished config {config_id} for environment {environment_to_string(environment)} in {result['time']} seconds")

        finally:
            queue.task_done()


def execute_all_configurations_old(config_file, output_dir, ssh_hosts, count, environments):
    """
    #todo

    Parameters:
        config_file (str): The file containing the configurations to be executed.
        output_dir (str): The directory into which the results should be saved
        ssh_hosts (list): The list of shh host to use for executing the transfers.
    """
    # read the configurations to be executed
    df = pd.read_csv(config_file)
    configs = df.to_dict(orient="records")[:count]  # converts dataframe to LIST of dicts

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    queue = Queue()

    for environment in environments:
        output_file = os.path.join(output_dir, f"{environment_to_string(environment)}_random_samples.csv")

        # check for already executed configurations
        if os.path.exists(output_file):
            df_out = pd.read_csv(output_file)
            if "config_id" in df_out.columns:
                executed_configs = df_out["config_id"].tolist()
            else:
                executed_configs = []
        else:
            executed_configs = []

        # filter configurations to skip already executed ones
        remaining_configs = [
            (i, config) for i, config in enumerate(configs) if i not in executed_configs
        ]

        if not remaining_configs:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] All configurations already executed for {environment_to_string(environment)} with N = {count}.")
            continue

        lock = threading.Lock()

        for config_id, config in remaining_configs:
            queue.put((config_id, config, environment, output_file, lock))



    # start threads
    threads = []
    for i in range(len(ssh_hosts)):
        thread = threading.Thread(target=process_configuration,
                                      args=(queue, ssh_hosts[i]))
        thread.start()
        threads.append(thread)

    # wait for threads to finish
    for thread in threads:
        thread.join()

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Completed execution for N={count}. Results saved to {output_file}.")

def execute_all_configurations(config_file, output_dir, ssh_hosts, count, environments):
    """
    #todo

    Parameters:
        config_file (str): The file containing the configurations to be executed.
        output_dir (str): The directory into which the results should be saved
        ssh_hosts (list): The list of shh host to use for executing the transfers.
    """


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    queue = Queue()

    for environment in environments:

        config_file = f"random_samples_{get_config_space_string(CONFIG_SPACE)}/configs/grid_configurations_{environment_to_string(environment)}_FIXED.csv"

        # read the configurations to be executed
        df = pd.read_csv(config_file)
        configs = df.to_dict(orient="records")[:count]  # converts dataframe to LIST of dicts


        output_file = os.path.join(output_dir, f"{environment_to_string(environment)}_random_samples.csv")

        # check for already executed configurations
        if os.path.exists(output_file):
            df_out = pd.read_csv(output_file)
            if "config_id" in df_out.columns:
                executed_configs = df_out["config_id"].tolist()
            else:
                executed_configs = []
        else:
            executed_configs = []

        # filter configurations to skip already executed ones
        remaining_configs = [
            (i, config) for i, config in enumerate(configs) if i not in executed_configs
        ]

        if not remaining_configs:
            print(f"[{datetime.today().strftime('%H:%M:%S')}] All configurations already executed for {environment_to_string(environment)} with N = {count}.")
            continue

        lock = threading.Lock()

        for config_id, config in remaining_configs:
            queue.put((config_id, config, environment, output_file, lock))



    # start threads
    threads = []
    for i in range(len(ssh_hosts)):
        thread = threading.Thread(target=process_configuration,
                                  args=(queue, ssh_hosts[i]))
        thread.start()
        threads.append(thread)

    # wait for threads to finish
    for thread in threads:
        thread.join()

    print(f"[{datetime.today().strftime('%H:%M:%S')}] Completed execution for N={count}. Results saved to {output_file}.")


def generate_random_configurations(n=1000,env=None):
    config_space_string = get_config_space_string(CONFIG_SPACE)

    filename = f"grid_configurations_{environment_to_string(env)}.csv"
    filepath = f"random_samples_{config_space_string}/"

    Path(filepath).mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame()
    first_write_done = False

    #optimizer = Syne_Tune_Ask_Tell(config_space=CONFIG_SPACE, underlying='grid_search')
    optimizer = LHS_Search_Optimizer(config_space=CONFIG_SPACE, n_samples=n)

    for i in range(0, n):

        suggested_config = optimizer.suggest()

        #optimizer.report(suggested_config, {'time': 1})

        suggested_config['config_id'] = i

        df = pd.DataFrame(suggested_config, index=[0])

        if not first_write_done:
            df.to_csv(filepath + filename, mode='a', header=True, index=False)
            first_write_done = True
        else:
            df.to_csv(filepath + filename, mode='a', header=False, index=False)

        results = pd.concat([results, df], axis=0)


CONFIG_SPACE = config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_EX_BufSiz

if __name__ == "__main__":
    '''
    for env in all_base_environments:
        generate_random_configurations(env=env)

    '''
    
    #generate_random_configurations()
    config_space_string = get_config_space_string(CONFIG_SPACE)


    output_dir = f"random_samples_{config_space_string}"

    ssh_hosts = reserved_hosts_big_cluster


    environments = [env_S16_C16_N1000]

    for i in [1000]:

        #for env in environments:

            #config_file = f"random_samples_{config_space_string}/configs/grid_configurations_{environment_to_string(env)}_FIXED.csv"
            print(f"starting executing with n = {i}")
            execute_all_configurations(None, output_dir, ssh_hosts, i, environments)

    #'''

    #for i in [450,
    #          #500, 600, 700, 800, 900,  1000
    #          ]:
    #    config_file = f"random_samples_{config_space_string}/grid_configurations_{environment_to_string(env)}_FIXED.csv"
    #    print(f"starting executing with n = {i}")
    #    execute_all_configurations(config_file, output_dir, ssh_hosts, i, environments)


