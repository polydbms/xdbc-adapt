from datetime import datetime

import paramiko
import json
import shlex
import time

from ssh_handler import SSHConnectionError
from Configs import *
from NestedSSHHandler import NestedSSHClient

REMOTE_PYTHON_EXEC = "python3"
REMOTE_GENERIC_WS_CLI_PATH = "~/ws_cli.py"
XDBC_CONTROLLER_WS_URI = "ws://localhost:8001"


def _execute_remote_websocket_command(ssh_client, operation, payload_data = None, ws_timeout = 30, get_pty=True, use_lc=False ):

    payload_str_arg = ""
    if payload_data is not None:
        payload_json_str = json.dumps(payload_data)
        payload_str_arg = f"--payload {shlex.quote(payload_json_str)}"

    command = (
        f"{REMOTE_PYTHON_EXEC} {REMOTE_GENERIC_WS_CLI_PATH} "
        f"--uri {XDBC_CONTROLLER_WS_URI} "
        f"--operation {shlex.quote(operation)} "
        f"{payload_str_arg} "
        f"--timeout {ws_timeout}"
    )

    if use_lc:
        command = f"bash -lc {shlex.quote(command)}"


    #print(f"[{datetime.today().strftime('%H:%M:%S')}] Executing remote command on {ssh_client.hostname}: {command}")


    try:
        output_str = ssh_client.execute_cmd(command, background=False, get_pty=get_pty)

        if not output_str:
            if operation == "start_transfer":
                return {"operation": "acknowledgement", "payload": "presumed_success_empty_output_start_transfer"}
            return {"status": "failure", "error": "Empty response from remote script execution. Check warnings for stderr.", "details": command}

        try:
            response_data = json.loads(output_str)
            #print(f"[WS_Manager_Nested] Parsed response: {json.dumps(response_data, indent=2)}")

            # Graceful handling for 'start_transfer' timeout
            if operation == "start_transfer" and \
                response_data.get("status") == "failure" and \
                response_data.get("error", "").startswith("Timeout: No response from WebSocket server"):
                return {"operation": "acknowledgement", "payload": "presumed_success_timeout_start_transfer"}

            #print(f"[{datetime.today().strftime('%H:%M:%S')}]  Parsed response: {json.dumps(response_data, indent=2)}")
            return response_data
        except json.JSONDecodeError as e:
            return {"status": "failure", "error": f"Failed to parse JSON response from remote script: {e}", "raw_response": output_str, "details": command}

    except SSHConnectionError as e:
        print(f"[{datetime.today().strftime('%H:%M:%S')}]  SSH Connection Error: {e}")
        return {"status": "failure", "error": f"SSH Connection Error: {str(e)}", "details": command}
    except Exception as e:

        print(f"[{datetime.today().strftime('%H:%M:%S')}] General execution error: {e}")
        return {"status": "failure", "error": f"General execution error: {str(e)}", "details": command}

def start_xdbc_transfer_parameter_wrapper(ssh_client, environment_config, client_config, server_config ):
    environment_config_payload = {
        "latency": "0",
        "packetLoss": "0",
        "maxBandwidth": f"{environment_config['network']}",
        "availableCoresClient":f"{environment_config['client_cpu']}",
        "availableCoresServer": f"{environment_config['server_cpu']}"
    }

    client_config_payload = {
        "table": f"{client_config['table']}",
        "intermediateFormat": f"{client_config['target_format']}",
        "bufferSize": f"{client_config['buffer_size']}",
        "bufferpoolSize": f"{client_config['client_buffpool_size']}",
        "netParallelism": f"{client_config['send_par']}",
        "writeParallelism": f"{client_config['write_par']}",
        "decompParallelism": f"{client_config['decomp_par']}",
        "serParallelism": f"{client_config['ser_par']}",
    }

    server_config_payload = {
        "system": f"{server_config['src']}",
        "compressionType": f"{server_config['compression']}",
        "intermediateFormat": f"{server_config['src_format']}",
        "bufferSize": f"{server_config['buffer_size']}",
        "bufferpoolSize": f"{server_config['server_buffpool_size']}",
        "readParallelism": f"{server_config['read_par']}",
        "readPartitions": "16",
        "deserParallelism": f"{server_config['deser_par']}",
        "netParallelism": f"{server_config['send_par']}",
        "compParallelism": f"{server_config['comp_par']}",
    }

    return start_xdbc_transfer(ssh_client,environment_config_payload,client_config_payload,server_config_payload)

def start_xdbc_transfer(ssh_client, environment_config, client_config, server_config ):
    results = {}
    #time.sleep(1)

    if server_config['compressionType'] == 'nocomp':
        server_config['compressionType'] = 'no_comp'

    # 1. Set environment
    print("Setting environment...")
    env_resp = _execute_remote_websocket_command(ssh_client, "set_environment", environment_config)
    results["set_environment"] = env_resp
    if not env_resp or env_resp.get("operation") == "error" or env_resp.get("status") == "failure":
        return {"status": "failure", "error": "Failed at set_environment", "details": results}
    #time.sleep(1)

    # 2. Set server parameters
    print("Setting server parameters...")
    server_resp = _execute_remote_websocket_command(ssh_client, "set_server_parameters", server_config)
    results["set_server_parameters"] = server_resp
    if not server_resp or server_resp.get("operation") == "error" or server_resp.get("status") == "failure":
        return {"status": "failure", "error": "Failed at set_server_parameters", "details": results}
    #time.sleep(1)

    # 3. Set client parameters
    print("Setting client parameters...")
    client_resp = _execute_remote_websocket_command(ssh_client, "set_client_parameters", client_config)
    results["set_client_parameters"] = client_resp
    if not client_resp or client_resp.get("operation") == "error" or client_resp.get("status") == "failure":
        return {"status": "failure", "error": "Failed at set_client_parameters", "details": results}

    #time.sleep(1)

    # 4. Start transfer
    print("Initiating start_transfer...")
    start_resp = _execute_remote_websocket_command(ssh_client, "start_transfer", payload_data=None, ws_timeout=5, get_pty=False, use_lc=True) # start might take longer
    results["start_transfer"] = start_resp
    if not start_resp or start_resp.get("operation") == "error" or start_resp.get("status") == "failure":
        return {"status": "failure", "error": "Failed at start_transfer", "details": results}
    #time.sleep(1)

    return {"status": "success", "message": "Transfer start sequence initiated", "details": results}


def get_xdbc_metrics_wrapper(ssh_client):
    metrics = get_xdbc_metrics(ssh_client)
    return (metrics['client_metrics']['payload']['progress_percentage'],metrics['client_metrics']['payload']['timestamp'])

def get_xdbc_metrics(ssh_client):
    #print(f"[{datetime.today().strftime('%H:%M:%S')}] Getting metrics")
    client_metrics_resp = _execute_remote_websocket_command(ssh_client, "get_client_metrics")
    server_metrics_resp = _execute_remote_websocket_command(ssh_client, "get_server_metrics")
    return {"client_metrics": client_metrics_resp, "server_metrics": server_metrics_resp}


def reconfigure_xdbc_transfer_wrapper(ssh_client, new_client_config, new_server_config):
    CLIENT_PARAMETERS_PAYLOAD = {
        "writeParallelism": f"{new_client_config['write_par']}",
        "decompParallelism": f"{new_client_config['decomp_par']}",
        "serParallelism": f"{new_client_config['ser_par']}",
    }

    if new_server_config['compression'] == 'nocomp':
        new_server_config['compression'] = 'no_comp'

    SERVER_PARAMETERS_PAYLOAD = {
        "compressionType": new_server_config['compression'],

        "readParallelism": f"{new_server_config['read_par']}",
        "deserParallelism": f"{new_server_config['deser_par']}",
        "compParallelism": f"{new_server_config['comp_par']}",
    }

    return reconfigure_xdbc_transfer(ssh_client,CLIENT_PARAMETERS_PAYLOAD,SERVER_PARAMETERS_PAYLOAD)
def reconfigure_xdbc_transfer(ssh_client, new_client_config, new_server_config):
    print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfiguring transfer")
    results = {}
    overall_status = "success"

    if new_client_config is not None:
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Setting new client parameters")
        client_resp = _execute_remote_websocket_command(ssh_client, "set_client_parameters", new_client_config)
        results["set_client_parameters"] = client_resp
        if not client_resp or client_resp.get("operation") == "error" or client_resp.get("status") == "failure":
            overall_status = "partial_failure"

    if new_server_config is not None:
        #print(f"[{datetime.today().strftime('%H:%M:%S')}] Setting new server parameters")
        server_resp = _execute_remote_websocket_command(ssh_client, "set_server_parameters", new_server_config)
        results["set_server_parameters"] = server_resp
        if not server_resp or server_resp.get("operation") == "error" or server_resp.get("status") == "failure":
            overall_status = "failure" if overall_status == "partial_failure" else "partial_failure"

    if new_client_config is None and new_server_config is None:
        return {"status": "no_action", "message": "No new configurations provided."}

    return {"status": overall_status, "details": results}


if __name__ == "__main__":
    SSH_HOST = "sr630-wn-a-20"


    DEFAULT_ENVIRONMENT_PAYLOAD = {
        "latency": "0",
        "packetLoss": "0",
        "maxBandwidth": "150",
        "availableCoresClient": "2",
        "availableCoresServer": "8"
    }
    DEFAULT_CLIENT_PARAMETERS_PAYLOAD = {
        "table": "lineitem_sf10",
        "intermediateFormat": "1",
        "bufferSize": "256",
        "bufferpoolSize": "16384",
        "netParallelism": "1",
        "writeParallelism": "1",
        "decompParallelism": "1",
        "serParallelism": "1",
    }
    DEFAULT_SERVER_PARAMETERS_PAYLOAD = {
        "system": "csv",
        "compressionType": "no_comp",
        "intermediateFormat": "1",
        "bufferSize": "256",
        "bufferpoolSize": "16384",
        "readParallelism": "1",
        "readPartitions": "1",
        "deserParallelism": "1",
        "netParallelism": "1",
        "compParallelism": "1",
    }

    ssh = NestedSSHClient(jump_host=big_cluster_main_host,
                          jump_username=get_username_for_host(big_cluster_main_host),
                          target_host=SSH_HOST,
                          target_username=get_username_for_host(SSH_HOST))

    try:



        # 1. Start transfer
        start_result = start_xdbc_transfer(
            ssh, DEFAULT_ENVIRONMENT_PAYLOAD, DEFAULT_CLIENT_PARAMETERS_PAYLOAD, DEFAULT_SERVER_PARAMETERS_PAYLOAD
        )
        time.sleep(3)
        start_timestamp = datetime.now()
        previous_percentage = None
        previous_timestamp = None
        print(f"[{datetime.today().strftime('%H:%M:%S')}] Start Result: {json.dumps(start_result, indent=2)}")

        if start_result.get("status") == "success":

            tranfer_running = True

            while(tranfer_running):

                time.sleep(5)

                # 2. Get metrics
                metrics = get_xdbc_metrics(ssh)
                current_timestamp = datetime.now()
                current_percentage = metrics['client_metrics']['payload']['progress_percentage']
                #print(f"Metrics: {json.dumps(metrics, indent=2)}")

                if metrics['client_metrics']['payload']['progress_percentage'] == 100:
                    tranfer_running = False
                    print(f"[{datetime.today().strftime('%H:%M:%S')}] Transfer finished, progress at 100%")
                    end_timestamp = datetime.now()

                    final_throughput = (7715741636 * metrics['client_metrics']['payload']['progress_percentage'] / 100) / 1000 / 1000 / (end_timestamp - start_timestamp).total_seconds()
                    total_seconds = (end_timestamp - start_timestamp).total_seconds()
                    print(f"[{datetime.today().strftime('%H:%M:%S')}] Finished transfer in {total_seconds} seconds, achieved average throughput of {final_throughput} MB/s")
                    break
                else:
                    print(f"[{datetime.today().strftime('%H:%M:%S')}] Transfer progresss at {metrics['client_metrics']['payload']['progress_percentage']}%")
                    throughput = (7715741636 * metrics['client_metrics']['payload']['progress_percentage'] / 100) / 1000 / 1000 / (datetime.now() - start_timestamp).total_seconds()
                    print(f"[{datetime.today().strftime('%H:%M:%S')}] Current transfer throughput since start : {throughput} MB/s")

                    if previous_percentage is not None:
                        throughput_since_last = (7715741636 * (current_percentage - previous_percentage) / 100) / 1_000_000 / ((current_timestamp - previous_timestamp).total_seconds())
                        print(f"[{datetime.today().strftime('%H:%M:%S')}] Current transfer throughput since last : {throughput_since_last} MB/s")


                    previous_timestamp = current_timestamp
                    previous_percentage = current_percentage


            # 3. Reconfigure
                #'''
                print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfiguring transfer")
                updated_client_config = DEFAULT_CLIENT_PARAMETERS_PAYLOAD.copy()
                updated_client_config["writeParallelism"] = "8"
                updated_client_config["serParallelism"] = "8"

                updated_server_config = DEFAULT_SERVER_PARAMETERS_PAYLOAD.copy()
                updated_server_config["readParallelism"] = "8"
                updated_server_config["compParallelism"] = "8"

                reconfig_result = reconfigure_xdbc_transfer(
                    ssh,
                    new_client_config=updated_client_config,
                    new_server_config=updated_server_config
                )
                #print(f"[{datetime.today().strftime('%H:%M:%S')}] Reconfig Result: {json.dumps(reconfig_result, indent=2)}")

                if reconfig_result.get("status") in ["success", "partial_failure"]:
                    #time.sleep(2)
                    metrics_after = get_xdbc_metrics(ssh)
                    #print(f"[{datetime.today().strftime('%H:%M:%S')}] Metrics after reconfig: {json.dumps(metrics_after, indent=2)}")
                #'''
        else:
            print("Start transfer failed. See details above.")





    except Exception as e:
        print(f"An unexpected local error occurred: {e}")
    finally:
        ssh.close()
        print("SSH Connection closed.")