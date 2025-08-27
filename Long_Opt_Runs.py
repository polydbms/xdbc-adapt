import os
from datetime import datetime
import random
import threading
import time
from queue import Queue

import pandas as pd

import data_transfer_wrapper
from Configs import *
from NestedSSHHandler import NestedSSHClient
from environments import *
from model_implementations.openbox_ask_tell import OpenBox_Ask_Tell


def queue_function(queue,ssh_host):
    while not queue.empty():
        environment, file, config_space, metric, mode, loop_count = queue.get()
        try:

            # create optimizer
            optimizer = OpenBox_Ask_Tell(config_space=config_space, metric=metric, mode=mode, underlying='bayesian_open_box')

            ssh = NestedSSHClient(jump_host=big_cluster_main_host,
                                  jump_username=get_username_for_host(big_cluster_main_host) ,
                                  target_host=ssh_host,
                                  target_username=get_username_for_host(ssh_host))

            i = 0
            if os.path.isfile(file):
                # load prevouis entries
                prevouis_evaluations = pd.read_csv(file)
                prevouis_evaluations = prevouis_evaluations[prevouis_evaluations['transfer_id'] > 0]
                i = len(prevouis_evaluations)
                for index, row in prevouis_evaluations.iterrows():
                    thread_count_client = row['write_par'] + row['ser_par'] + row['decomp_par'] + row['rcv_par']
                    client_bufpool_factor = row['client_bufferpool_size'] / row['buffer_size'] / 2 / thread_count_client


                    thread_count_server = row['read_par'] + row['deser_par'] + row['comp_par'] + row['send_par']
                    server_bufpool_factor = row['server_bufferpool_size'] / row['buffer_size'] / 2 / thread_count_server


                    config = {
                        "compression": row["compression"],
                        "format": row["format"],
                        "client_bufpool_factor": client_bufpool_factor,
                        "server_bufpool_factor": server_bufpool_factor,
                        "buffer_size": row["buffer_size"],
                        "send_par": row["send_par"],
                        "write_par": row["write_par"],
                        "decomp_par": row["decomp_par"],
                        "read_par": row["read_par"],
                        "deser_par": row["deser_par"],
                        "ser_par": row["ser_par"],
                        "comp_par": row["comp_par"],
                    }
                    result = {metric: row[metric]}
                    print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh.hostname}] [{optimizer.underlying}] loaded prevouis result for environment {environment_to_string(environment)}")
                    optimizer.report(config, result)


            # run opt run

            results = pd.DataFrame()
            time_lost_too_timeouts = 0

            while i < loop_count+1:

                print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh.hostname}] [{optimizer.underlying}] starting  transfer #{i} for environment {environment_to_string(environment)}")

                suggested_config = optimizer.suggest()
                complete_config = create_complete_config(environment, metric, 'dict', suggested_config)
                result = None

                result = data_transfer_wrapper.transfer(complete_config, i, max_retries=0, ssh=ssh)

                if result['transfer_id'] == -1:
                    time_lost_too_timeouts = 0
                    #try once more
                    start = datetime.now()

                    result = data_transfer_wrapper.transfer(complete_config, i, max_retries=0, ssh=ssh)
                    if result['transfer_id'] == -1:
                        time_lost_too_timeouts = 0
                        loop_count = loop_count+1
                        # if failed twice, add another iteration to still get to the specified number of evaluations for that algorithm

                optimizer.report(suggested_config, result)

                result['trial_id'] = -1
                result['algo'] = optimizer.underlying
                result['seconds_since_start_of_opt_run'] = i

                df = pd.DataFrame(result, index=[0])

                if os.path.isfile(file):
                    df.to_csv(file, mode='a', header=False, index=False)
                else:
                    df.to_csv(file, mode='a', header=True, index=False)

                results = pd.concat([results, df], axis=0)
                print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh.hostname}] [{optimizer.underlying}] completed transfer #{i} for environment {environment_to_string(environment)} (result was {result[metric]})")
                i += 1

        finally:
            ssh.close()
            queue.task_done()
            time.sleep(random.uniform(0, 1)*5)
            print(f"\n\n[{datetime.today().strftime('%H:%M:%S')}] Finished run for {environment_to_string(environment)}\n\n")

    print(f"\n\n[{datetime.today().strftime('%H:%M:%S')}] [{ssh_host}] Thread finished execution, Queue is empty.\n\n")


def start_runs():
    config_space = config_space_variable_parameters_generalized_FOR_NEW_ITERATION_FLEXIBLE_DISCRETE_EX_BufSiz

    queue = Queue()



    # fill queue
    for env in all_base_environments:
        # env, file, config_space, metric, mode, loop_count

        file = os.path.join('long_bayesian_runs_PG_PD', f"{environment_to_string(env)}_long_bayesian_run.csv")

        queue.put((env, file, config_space, 'uncompressed_throughput', 'max', 1000))

    # start workers



    ssh_hosts = [   "sr630-wn-a-31",
                    "sr630-wn-a-32",
                    "sr630-wn-a-33",
                    "sr630-wn-a-34",
                    "sr630-wn-a-35",
                    "sr630-wn-a-36",
                    "sr630-wn-a-37",
                    "sr630-wn-a-38",
                    "sr630-wn-a-39",
                    "sr630-wn-a-40",
                    "sr630-wn-a-41",
                    "sr630-wn-a-42",
                    "sr630-wn-a-43",
                    "sr630-wn-a-44",
                    "sr630-wn-a-45",
                    "sr630-wn-a-46",
                    "sr630-wn-a-47",
                    "sr630-wn-a-48", ]

    threads = []
    for i in range(len(ssh_hosts)):
        thread = threading.Thread(target=queue_function, args=(queue, ssh_hosts[i]))
        thread.start()
        threads.append(thread)
        time.sleep(15)  # so no threads start at the same time, bc filenames depend on timestamp


    # wait for threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":

    not_finished = True
    while not_finished:
        try:
            start_runs()

            not_finished = False

        except:
            print("Runs crashed, retrying in 60 seconds")
            time.sleep(60)
            print("Now starting again")
