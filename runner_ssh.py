import subprocess
import threading
import time
import datetime
import csv
import os

from ssh_handler import SSHConnection
from metrics_client import MetricsClient
from metrics_server import MetricsServer

def run_xdbserver_and_xdbclient(config, env, mode, perf_dir, ssh, return_transfer_id=False, sleep=2, show_output=(False, False),include_setup=True):


    if include_setup:
        ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml down")
        ssh.execute_cmd("docker compose -f xdbc-client/docker-xdbc.yml up -d")
        ssh.execute_cmd("docker exec xdbcclient bash -c \"ldconfig\"")
        ssh.execute_cmd('docker exec xdbcserver bash -c "[ ! -f /tmp/xdbc_server_timings.csv ] && echo \'transfer_id,total_time,read_wait_time,read_proc_time,read_throughput,read_throughput_pb,free_load,deser_wait_time,deser_proc_time,deser_throughput,deser_throughput_pb,deser_load,comp_wait_time,comp_proc_time,comp_throughput,comp_throughput_pb,comp_load,send_wait_time,send_proc_time,send_throughput,send_throughput_pb,send_load\' > /tmp/xdbc_server_timings.csv"')
        ssh.execute_cmd('docker exec xdbcclient bash -c "[ ! -f /tmp/xdbc_client_timings.csv ] && echo \'transfer_id,total_time,rcv_wait_time,rcv_proc_time,rcv_throughput,rcv_throughput_pb,free_load,decomp_wait_time,decomp_proc_time,decomp_throughput,decomp_throughput_pb,decomp_load,ser_wait_time,ser_proc_time,ser_throughput,ser_throughput_pb,ser_load,write_wait_time,write_proc_time,write_throughput,write_throughput_pb,write_load\' > /tmp/xdbc_client_timings.csv"')




    show_server_output, show_client_output = show_output
    show_stdout_server = None if show_server_output else subprocess.DEVNULL
    show_stdout_client = None if show_client_output else subprocess.DEVNULL

    # subprocess.run([f"curl -X DELETE localhost:4080/xdbcserver && curl -X PUT localhost:4080/xdbcserver"])


    ssh.execute_cmd(f"docker update --cpus '{env['server_cpu']}' {env['server_container']}")
    ssh.execute_cmd(f"docker update --cpus '{env['client_cpu']}' {env['client_container']}")

    ssh.execute_cmd(f"curl -X DELETE localhost:4080/{env['client_container']}")
    ssh.execute_cmd(f"curl -X PUT localhost:4080/{env['client_container']}")

    if env['network'] != 0:
        ssh.execute_cmd(f"curl -s -d rate={env['network']}mbps localhost:4080/{env['client_container']}")



    server_path =   os.path.abspath(os.path.join(os.getcwd(), '..', 'xdbc-server', 'experiments'))
    client_path =  "~/xdbc-client/experiments"

    #1 / 3
    measurement_path = os.path.abspath(os.path.join(perf_dir, 'xdbc_general_stats_bene.csv'))
    #measurement_path = os.path.abspath(os.path.join(perf_dir, 'xdbc_general_stats.csv'))

    config['host'] = ssh.hostname

    config['client_readmode'] = mode
    config['read_partitions'] = 1

    if 'format' not in config.keys():
        config['format'] = config['src_format']

    config['skip_ser'] = 0# ??? todo !!!!!

    config['run'] = 1
    config['date'] = int(time.time_ns())
    config['transfer_id'] = int(time.time_ns() + hash(ssh.hostname)%100000000000)
    config['xdbc_version'] = 10
    config['system_source'] = env["src"]
    config['system_dest'] = env["target"]

    #if 'compression_lib' not in config:
    #    config['compression_lib'] = "nocomp"

    #if 'ser_par' not in config:
    #    config['ser_par'] = 4

    result = {}
    profiling_interval = 1000

    #print("----------------------------------------")
    #print("XDBC SSH-Runner with config:")
    #print(config)

    try:

        command_server = (f"docker exec -d xdbcserver bash -c " +
                          f""" \" ./xdbc-server/build/xdbc-server \
                         --network-parallelism={config['send_par']} \
                         --read-partitions=1 \
                         --read-parallelism={config['read_par']} \
                          -c{config['compression_lib']} \
                          --compression-parallelism={config['comp_par']} \
                          --buffer-size={config['buffer_size']} \
                          --dp={config['deser_par']} \
                          -p{config['server_buffpool_size']} \
                          -f{config['format']} \
                          --skip-deserializer={config['skip_ser']} \
                          --tid="{config['transfer_id']}" \
                          --system={env['src']} \
                          --profiling-interval={profiling_interval}"
                        """)
        ssh.execute_cmd(command_server, True)

        time.sleep(sleep)

        start_data_size = measure_network(client_path, env['client_container'], ssh)
        a = datetime.datetime.now()

        if env['target'] == 'csv':
            '''
            command_client = f"""docker exec -it {env['client_container']} bash -c \" ./xdbc-client/tests/build/test_xclient \
                                 --server-host={env['server_container']} \
                                --table="{env['table']}" \
                                -f{config['format']} \
                                -b{config['buffer_size']} \
                                -p{config['client_buffpool_size']} \
                                -n{config['rcv_par']} \
                                -r{config['write_par']} \
                                -d{config['decomp_par']} \
                                -s1 \
                                --tid="{config['date']}" \
                                -m{mode} \"
                            """
            '''
            command_client = f"""docker exec -it {env['client_container']} bash -c \" ./xdbc-client/Sinks/build/xdbcsinks \
                                 --server-host={env['server_container']} \
                                --table="{env['table']}" \
                                -f{config['format']} \
                                -b{config['buffer_size']} \
                                -p{config['client_buffpool_size']} \
                                -n{config['rcv_par']} \
                                -w{config['write_par']} \
                                -d{config['decomp_par']} \
                                -s{config['ser_par']} \
                                --skip-serializer={config['skip_ser']} \
                                --target={env['target']} \
                                --tid="{config['transfer_id']}" \
                                --profiling-interval={profiling_interval} \"
                                
                            """
            ssh.execute_cmd(command_client)
            #print("nn")


        elif env['target'] == 'pandas':
            print("Running pandas")

            command_pandas_client = f"docker exec -it {env['client_container']} bash -c \" python /workspace/tests/pandas_xdbc.py \
                            --env_name \\\"PyXDBC Client\\\" \
                            --table \\\"{env['table']}\\\" \
                            --iformat {config['format']} \
                            --buffer_size {config['buffer_size']} \
                            --bufferpool_size {config['client_buffpool_size']} \
                            --sleep_time 1 \
                            --rcv_par {config['rcv_par']} \
                            --ser_par {config['write_par']} \
                            --write_par {config['write_par']} \
                            --decomp_par {config['decomp_par']} \
                            --transfer_id {config['date']} \
                            --server_host {env['server_container']} \
                            --server_port \\\"1234\\\" \
                            --source {env['src']} \
                            --skip_ser {config['skip_ser']} \""

            

            ssh.execute_cmd(command_pandas_client)
            #print('XX')

        elif env['target'] == 'spark':
            print("Running spark")
            ssh.execute_cmd(["docker", "exec", "-it", env['client_container'], "bash", "-c",
                            f"""bash run_xdbc_spark.sh \
                            tableName={env['table']} \
                            iformat={config['format']} \
                            buffer_size={config['buffer_size']} \
                            bufferpool_size={config['client_buffpool_size']} \
                            rcv_par={config['rcv_par']} \
                            write_par={config['write_par']} \
                            decomp_par={config['decomp_par']} \
                            transfer_id={config['date']} \
                            server_host={env['server_container']}
                            """])

        elif env['target'] == 'postgres':
            print("Running postgres")

            subprocess.run(f"docker exec -it {env['client_container']} bash -c \" cd /pg_xdbc_fdw/experiments/ && bash replace_options.sh -t {env['table']} -s {env['server_container']} -i {config['date']} -b {config['buffer_size']} -p {config['client_buffpool_size']} -n {config['rcv_par']} -d {config['decomp_par']} -r {config['write_par']} \"",)
            
            subprocess.run(f"docker exec -it {env['client_container']} bash -c \" bash /pg_xdbc_fdw/experiments/run_pg_xdbc.sh {env['table']} {config['date']} \"")

        b = datetime.datetime.now()
        c = b - a
        result['time'] = c.total_seconds()

        end_data_size = measure_network(client_path, env['client_container'],ssh)
        result['size'] = end_data_size - start_data_size
        result['avg_cpu_server'] = 0
        result['avg_cpu_client'] = 0
        result['transfer_id'] = config['transfer_id']

        #print(f"Total Data Transfer Size: {result['size']}")
        #write_csv_header(measurement_path)
        config['date'] = config['transfer_id']
        write_to_csv(measurement_path, env, config, result)
        copy_metrics(env['server_container'], env['client_container'], perf_dir,ssh)

        res = result['time']
    except subprocess.CalledProcessError as e:
        print(f"Error running XDBC: {e}")
        res = -1

    pkill_cmd_str = f"docker exec {env['server_container']} bash -c pkill -f './xdbc-server/build/xdbc-server'"
    ssh.execute_cmd(pkill_cmd_str, True)


    if env['target'] == 'csv':

        target_kill_str = f"docker exec {env['client_container']} bash -c rm -rf /dev/shm/output*"
        ssh.execute_cmd(target_kill_str,True)

    if return_transfer_id:
        return result
    else:
        return res


def check_file_exists(container, file_path, ssh):
    result = ssh.execute_cmd(
        f"docker exec {container} test -f {file_path}"
    )
    rc = 0
    if result != None:
        rc = 1
    return rc


def copy_metrics(server_container, client_container, perf_dir,ssh):
    file_path_server = '/tmp/xdbc_server_timings.csv'
    file_path_client = '/tmp/xdbc_client_timings.csv'

    if (not check_file_exists(server_container, file_path_server,ssh) or
            not check_file_exists(client_container, file_path_client,ssh)):
        return False

    absolute_perf_dir = os.path.join(os.getcwd(), perf_dir)
    os.makedirs(absolute_perf_dir, exist_ok=True)

    try:

        data_server = ssh.execute_cmd(f"docker exec {server_container} tail -n 1 {file_path_server}")
        data_client = ssh.execute_cmd(f"docker exec {client_container} tail -n 1 {file_path_client}")
        file_lock = threading.Lock()

        if data_server.startswith("transfer"):
            print(f"\n\n [{ssh.hostname}] found header in server data file\n\n")

            raise ValueError(f"\n\n\n\n\n\n\n [{ssh.hostname}] found header in server data file\n\n\n\n\n\n\n")


        if data_client.startswith("transfer"):
            print(f"\n\n [{ssh.hostname}] found header in client data file\n\n")

            raise ValueError(f"\n\n\n\n\n\n\n [{ssh.hostname}] found header in client data file\n\n\n\n\n\n\n")

        with file_lock:
            #2 / 3

            with open(f"{absolute_perf_dir}/xdbc_server_timings_bene.csv",'a') as file_server:
            #with open(f"{absolute_perf_dir}/xdbc_server_timings.csv",'a') as file_server:

                if "cannot open" in data_server:
                    print("error reading timing file")
                else:
                    file_server.write("\n")
                    file_server.write(data_server)

            #3 / 3

            with open(f"{absolute_perf_dir}/xdbc_client_timings_bene.csv",'a') as file_client:
            #with open(f"{absolute_perf_dir}/xdbc_client_timings.csv",'a') as file_client:
            # todo change back

                if "cannot open" in data_client:
                    print("error reading timing file")
                else:
                    file_client.write("\n")
                    file_client.write(data_client)

        #x = print_metrics(perf_dir, True)


    except subprocess.CalledProcessError as e:
        print(f"Error during file copy: {e}")
        return False

    return True


def print_metrics(perf_dir, dict=False):
    absolute_perf_dir = os.path.join(os.getcwd(), perf_dir)

    metrics_server = MetricsServer.from_csv(f"{absolute_perf_dir}/xdbc_server_timings_bene.csv")
    metrics_client = MetricsClient.from_csv(f"{absolute_perf_dir}/xdbc_client_timings_bene.csv")
    if dict:
        return {**metrics_server.get_throughput_metrics(False),
                **metrics_client.get_throughput_metrics(False)}
    print(metrics_server.get_throughput_metrics())
    print(metrics_client.get_throughput_metrics())


def measure_network(client_path, container, ssh):
    result = ssh.execute_cmd(
        f"bash {client_path}/experiments_measure_network.sh '{container}' "
    )
    return int(result)

'''
def write_csv_header(filename, config=None):
    header = ['date', 'xdbc_version', 'host', 'run', 'source_system', 'target_system', 'table', 'compression',
              'format', 'send_par', 'rcv_par', 'server_bufferpool_size', 'client_bufferpool_size', 'buffer_size',
              'network', 'network_latency', 'network_loss', 'client_readmode', 'client_cpu', 'write_par', 'decomp_par',
              'server_cpu', 'read_par', 'read_partitions', 'deser_par',#'ser_par',
              'comp_par', 'time', 'datasize',
              'avg_cpu_server', 'avg_cpu_client']

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
'''

def write_to_csv(filename, env, config, result):
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [config['date']] + [config['xdbc_version'], config['host']] +
            [config['run'], env['src'], env['target'], env['table'],
             config['compression_lib'],
             config['format'], config['send_par'], config['rcv_par'],
             config['server_buffpool_size'], config['client_buffpool_size'],
             config['buffer_size'], env['network'], env['network_latency'], env['network_loss'],
             config['client_readmode'],
             env['client_cpu'], config['write_par'],
             config['decomp_par'],
             env['server_cpu'], config['read_par'],
             config['read_partitions'],
             config['deser_par'],
             config['ser_par'],
             config['comp_par']] +
            [result['time'], result['size'], result['avg_cpu_server'], result['avg_cpu_client']])
