import threading
import warnings

import paramiko

from ssh_handler import SSHConnectionError
from Configs import reserved_hosts_big_cluster


class NestedSSHClient:
    def __init__(self, jump_host, jump_username, target_host, target_username):
        self.jump_host = jump_host
        self.jump_username = jump_username
        self.target_host = target_host
        self.target_username = target_username

        self.hostname = target_host

        self.jump_client = None
        self.target_client = None

        self._connect()

    global_lock = threading.Lock()

    def _connect(self):
        with NestedSSHClient.global_lock:
            self.jump_client = paramiko.SSHClient()
            self.jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.jump_client.connect(
                self.jump_host,
                username=self.jump_username,
                allow_agent=True,
                look_for_keys=True,
            )

            transport = self.jump_client.get_transport()
            dest_addr = (self.target_host, 22)
            local_addr = ('127.0.0.1', 0)
            channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
            transport.set_keepalive(60)
            #todo !!!
            self.target_client = paramiko.SSHClient()
            self.target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.target_client.connect(
                self.target_host,
                username=self.target_username,
                sock=channel,
                allow_agent=True,
                look_for_keys=True,

            )
            self.target_client.get_transport().set_keepalive(60)

    def execute_cmd(self, cmd, background=False, get_pty=True):

        if not self.target_client.get_transport() or not self.target_client.get_transport().is_active():
            raise SSHConnectionError(f"SSH connection is not active for {self.hostname}: ${cmd}", self.hostname)


        if background:
            background_command = f"nohup {cmd} > /dev/null 2>&1 &"
            self.target_client.exec_command(background_command)
            return None
        else:
            stdin, stdout, stderr = self.target_client.exec_command(cmd, get_pty=get_pty)
            error_output = stderr.read().decode().strip()
            output_from_stdout = stdout.read().decode().strip()

            if error_output:
                warnings.warn(f"error on host: {self.hostname}, cmd: {cmd}, error: {error_output}")
                if output_from_stdout:
                    return output_from_stdout
            return output_from_stdout#stdout.read().decode().strip()


    def close(self):

        if self.target_client:
            self.target_client.close()
        if self.jump_client:
            self.jump_client.close()



if __name__ == "__main__":


    #executed for 26 - 46
    #ssh.execute_cmd("make -C ./xdbc-client/.")
    #ssh.execute_cmd("make -C ./xdbc-server/.")



    '''

    jump_host = "sr630-wn-a-01.dima.tu-berlin.de"
    jump_username = "bdidrich-ldap"

    target_host = "sr630-wn-a-42"
    target_username = "bdidrich-ldap"

    ssh = NestedSSHClient(
        jump_host, jump_username,
        target_host, target_username
    )

    stdout = ssh.execute_cmd("hostname")

    print("Output:", stdout)

    ssh.close()

    '''



