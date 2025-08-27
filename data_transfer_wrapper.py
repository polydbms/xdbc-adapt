import datetime
import os
import threading

import pandas as pd
from ssh_handler import SSHConnection
from Metrics import add_all_metrics_to_result
from datetime import datetime
from func_timeout import func_timeout, FunctionTimedOut

from Configs import get_username_for_host, all_hosts_cloud_7
import runner_ssh


def transfer(config,i=0,max_retries=1, ssh=None):
    """
    Description of Funtion. #todo

    Parameters:
        param_1 (datatype): description.
        param_2 (datatype): description.

    Returns:
        datatype: descritpion.
    """

    if config['timeout'] is None:
        config['timeout'] = 500



    config["rcv_par"] = config["send_par"]


    if "client_bufpool_factor" in config.keys():
        thread_count_client = config['write_par'] + config['ser_par'] + config['decomp_par'] + config['rcv_par']
        min_buffer_count_client = thread_count_client * 2
        config["client_buffpool_size"] = min_buffer_count_client * config["client_bufpool_factor"] * config['buffer_size']

    if "server_bufpool_factor" in config.keys():
        thread_count_server = config['read_par'] + config['deser_par'] + config['comp_par'] + config['send_par']
        min_buffer_count_server = thread_count_server * 2
        config["server_buffpool_size"] = min_buffer_count_server * config["server_bufpool_factor"] * config['buffer_size']


    if 'format' in config.keys():
        config['src_format'] = config['format']
        config['target_format'] = config['format']

    if 'compression' in config.keys():
        config['compression_lib'] = config['compression']

    if 'ser_par' not in config.keys():
        config['ser_par'] = 1



    result = train_method(config, ssh)
    retries = 0

    while not result and retries < max_retries:
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh.hostname}] transfer #{i} failed for {retries + 1} times, retrying for {retries + 1} time")
        result = train_method(config, ssh)
        retries = retries + 1

    if not result:
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh.hostname}] transfer #{i} failed for {retries + 1} times, reporting dummy result")

        try:
            save_failed_config(config)
        except:
            pass
        '''
        dummy_result = {
            "transfer_id": -1,
            "total_time_x": -1,
            "rcv_wait_time": -1,
            "rcv_proc_time": -1,
            "rcv_throughput": -1,
            "rcv_throughput_pb": -1,
            "rcv_load": -1,
            "decomp_wait_time": -1,
            "decomp_proc_time": -1,
            "decomp_throughput": -1,
            "decomp_throughput_pb": -1,
            "decomp_load": -1,
            "write_wait_time": -1,
            "write_proc_time": -1,
            "write_throughput": -1,
            "write_throughput_pb": -1,
            "write_load": -1,
            "total_time_y": -1,
            "read_wait_time": -1,
            "read_proc_time": -1,
            "read_throughput": -1,
            "read_throughput_pb": -1,
            "read_load": -1,
            "deser_wait_time": -1,
            "deser_proc_time": -1,
            "deser_throughput": -1,
            "deser_throughput_pb": -1,
            "deser_load": -1,
            "comp_wait_time": -1,
            "comp_proc_time": -1,
            "comp_throughput": -1,
            "comp_throughput_pb": -1,
            "comp_load": -1,
            "send_wait_time": -1,
            "send_proc_time": -1,
            "send_throughput": -1,
            "send_throughput_pb": -1,
            "send_load": -1,
            "date": -1,
            "xdbc_version": -1,
            "host": ssh.hostname,
            "run": -1,
            "source_system": -1,
            "target_system": -1,
            "table": config["table"],
            "compression": config["compression_lib"],
            "format": -1,
            "send_par": config["send_par"],
            "rcv_par": -1,
            "server_bufferpool_size": -1,
            "client_bufferpool_size": -1,
            "buffer_size": config["buffer_size"],
            "network": config["network"],
            "network_latency": -1,
            "network_loss": -1,
            "client_readmode": -1,
            "client_cpu": config["client_cpu"],
            "write_par": config["write_par"],
            "decomp_par": config["decomp_par"],
            "server_cpu": config["server_cpu"],
            "read_par": config["read_par"],
            "read_partitions": 1,
            "deser_par": config["deser_par"],
            "comp_par": config["comp_par"],
            "ser_par": config["ser_par"],
            "time": config['timeout']*2, # todo test !!!! # idea : report a really bad result
            "datasize": -1,
            "avg_cpu_server": -1,
            "avg_cpu_client": -1,
            "timestamp_end": -1,
            "throughput": -1,
            "average_throughput": -1,
            "resource_utilization": -1,
            "wait_to_proc_time_ratio": -1,
            "even_load_distribution_mse": -1,
            "uncompressed_throughput": (7715741636 / 1000 / 1000) / config['timeout']*2
        }
        '''
        dummy_result={
            "transfer_id":-1,
            "total_time_x":-1,

            "rcv_wait_time":-1,
            "rcv_proc_time":-1,
            "rcv_throughput":-1,
            "rcv_throughput_pb":-1,

            "free_load_x":-1,

            "decomp_wait_time":-1,
            "decomp_proc_time":-1,
            "decomp_throughput":-1,
            "decomp_throughput_pb":-1,
            "decomp_load":-1,

            "ser_wait_time":-1,
            "ser_proc_time":-1,
            "ser_throughput":-1,
            "ser_throughput_pb":-1,
            "ser_load":-1,

            "write_wait_time":-1,
            "write_proc_time":-1,
            "write_throughput":-1,
            "write_throughput_pb":-1,
            "write_load":-1,

            "total_time_y":-1,

            "read_wait_time":-1,
            "read_proc_time":-1,
            "read_throughput":-1,
            "read_throughput_pb":-1,

            "free_load_y":-1,

            "deser_wait_time":-1,
            "deser_proc_time":-1,
            "deser_throughput":-1,
            "deser_throughput_pb":-1,
            "deser_load":-1,

            "comp_wait_time":-1,
            "comp_proc_time":-1,
            "comp_throughput":-1,
            "comp_throughput_pb":-1,
            "comp_load":-1,

            "send_wait_time":-1,
            "send_proc_time":-1,
            "send_throughput":-1,
            "send_throughput_pb":-1,
            "send_load":-1,

            "date":-1,
            "xdbc_version":-1,
            "host":ssh.hostname,
            "run":-1,

            "source_system":-1,
            "target_system":-1,
            "table":config["table"],
            "compression":config["compression"],
            "format":config['format'],
            "send_par":config['send_par'],
            "rcv_par":config['rcv_par'],
            "server_bufferpool_size":-1,
            "client_bufferpool_size":-1,
            "buffer_size":config['buffer_size'],
            "network":config['network'],
            "network_latency":-1,
            "network_loss":-1,
            "client_readmode":config['client_readmode'],
            "client_cpu":config['client_cpu'],
            "write_par":config['write_par'],
            "decomp_par":config['decomp_par'],
            "server_cpu":config['server_cpu'],
            "read_par":config['read_par'],
            "read_partitions":-1,
            "deser_par":config['deser_par'],
            "ser_par":config['ser_par'],
            "comp_par":config['comp_par'],
            "time":config['timeout']*2,
            "datasize":-1,
            "avg_cpu_server":-1,
            "avg_cpu_client":-1,
            "timestamp_end":-1,
            "throughput":-1,
            "average_throughput":-1,
            "resource_utilization":-1,
            "wait_to_proc_time_ratio":-1,
            "uncompressed_throughput":(7715741636 / 1000 / 1000) / config['timeout']*2,
            #"client_bufpool_factor":-1,
            #"server_bufpool_factor":-1
        }

        return dummy_result
    else:
        return result


def save_failed_config(config):
    """
    Description of Funtion. #todo

    Parameters:
        param_1 (datatype): description.
        param_2 (datatype): description.

    Returns:
        datatype: descritpion.
    """
    filename = f"failed_configs.csv"

    df = pd.DataFrame(config, index=[0])

    #df.to_csv(filename, mode='a', header=True)

    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='a', header=True, index=False)



def train_method(config_p,ssh):



    config = {
        # fixded / enviroment params
        "xdbc_version": config_p['xdbc_version'],
        "run": config_p['run'],
        "client_readmode": config_p['client_readmode'],
        "client_cpu": config_p['client_cpu'],
        "server_cpu": config_p['server_cpu'],
        "network": config_p['network'],
        "network_latency": config_p['network_latency'],
        "network_loss": config_p['network_loss'],
        "table": config_p['table'],
        'src': config_p['src'],
        'src_format': config_p['src_format'],
        'target': config_p['target'],
        'target_format': config_p['target_format'],
        'server_container': config_p['server_container'],
        'client_container': config_p['client_container'],

        # varying params
        "compression_lib": config_p['compression_lib'],

        "server_buffpool_size": config_p['server_buffpool_size'],
        "client_buffpool_size": config_p['client_buffpool_size'],

        "buffer_size": config_p['buffer_size'],

        "send_par":config_p['send_par'],
        "rcv_par":config_p['send_par'],

        "write_par":config_p['write_par'],
        "decomp_par":config_p['decomp_par'],
        #"server_read_partitions":config_p['read_partitions'],
        "read_par":config_p['read_par'],
        "deser_par":config_p['deser_par'],
        "ser_par":config_p['ser_par'],
        "comp_par":config_p['comp_par'],
    }


    timeout = config_p['timeout']



    ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml down")
    ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml up -d")
    ssh.execute_cmd("docker compose -f xdbc-client/docker-tc.yml up -d")
    ssh.execute_cmd("docker exec xdbcclient bash -c \"ldconfig\"")
    ssh.execute_cmd('docker exec xdbcserver bash -c "[ ! -f /tmp/xdbc_server_timings.csv ] && echo \'transfer_id,total_time,read_wait_time,read_time,deser_wait_time,deser_time,compression_wait_time,compression_time,network_wait_time,network_time\' > /tmp/xdbc_server_timings.csv"')


    ssh.execute_cmd('docker exec xdbcclient bash -c "[ ! -f /tmp/xdbc_client_timings.csv ] && echo \'transfer_id,total_time,rcv_wait_time,rcv_time,decomp_wait_time,decomp_time,write_wait_time,write_time\' > /tmp/xdbc_client_timings.csv"')


    try:
                                                                                             #(config, env, mode, perf_dir, ssh=None, return_transfer_id=False, sleep=2, show_output=(False, False),include_setup=True):
        result = func_timeout(timeout=timeout, func=runner_ssh.run_xdbserver_and_xdbclient, args=[config, config, config['client_readmode'], "local_measurements", ssh, True,2,(False,False),False])

        complete_result = load_complete_result(result)
        return complete_result


    except FunctionTimedOut:
        print(f"[{datetime.today().strftime('%H:%M:%S')}] [{ssh.hostname}] transfer  could not be completet within {timeout} seconds and was terminated")
        ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml down")
        return {}

    return result


def load_complete_result(result):
    """
    Description of Funtion. #todo

    Parameters:
        param_1 (datatype): description.
        param_2 (datatype): description.

    Returns:
        datatype: descritpion.
    """
    file_lock = threading.Lock()

    with file_lock:

        try:

            #client_timings = pd.read_csv("/optimizer/local_measurements/xdbc_client_timings_bene.csv")
            #server_timings = pd.read_csv("/optimizer/local_measurements/xdbc_server_timings_bene.csv")
            #general_stats = pd.read_csv("/optimizer/local_measurements/xdbc_general_stats_bene.csv")

            client_timings = pd.read_csv("local_measurements/xdbc_client_timings_bene.csv")
            server_timings = pd.read_csv("local_measurements/xdbc_server_timings_bene.csv")
            general_stats = pd.read_csv("local_measurements/xdbc_general_stats_bene.csv")




            df_both_timings = pd.merge(client_timings,server_timings,on='transfer_id')
            df_complete = pd.merge(df_both_timings,general_stats,left_on="transfer_id",right_on="date")
            df_result = df_complete[(df_complete.transfer_id == result['transfer_id'])]

            df_result['timestamp_end'] = str(datetime.now())
            df_result['transfer_id'] = df_result['transfer_id'].astype(str)
            result['transfer_id'] = str(result['transfer_id'])

            result = df_result.iloc[0].to_dict()
            result = add_all_metrics_to_result(result)
        except:
            print(f"failed to load data transfer result for transfer id {result['transfer_id']}")

        return result