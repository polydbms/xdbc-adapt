import paramiko
import time
import socket
import warnings
import threading


def simple_warning_format(message, category, filename, lineno, file=None, line=None):
    return f"{category.__name__}: {message}\n"


# Assign custom function to handle warning outputs
warnings.showwarning = simple_warning_format


class SSHExecutionWarning(UserWarning):
    """Raised when there's an error executing a command over SSH."""
    pass


class SSHConnectionError(Exception):
    """Raised when an SSH connection is not active or command execution fails."""

    def __init__(self, message, host=None):
        super().__init__(message)
        self.host = host


class SSHConnection:
    def __init__(self, host, username, password=""):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(host, username=username, password=password)
        self.ssh.get_transport().set_keepalive(60)
        self.lock = threading.Lock()
        ip_address = self.ssh.get_transport().getpeername()[0]
        self.hostname = socket.gethostbyaddr(ip_address)[0]

    def execute_cmd(self, cmd, background=False):

        if not self.ssh.get_transport() or not self.ssh.get_transport().is_active():
            raise SSHConnectionError(f"SSH connection is not active for {self.hostname}: ${cmd}", self.hostname)

        if background:
            background_command = f"nohup {cmd} > /dev/null 2>&1 &"
            self.ssh.exec_command(background_command)
            return None
        else:
            stdin, stdout, stderr = self.ssh.exec_command(cmd,get_pty=True)
            error_output = stderr.read().decode().strip()
            if error_output:
                warnings.warn(f"error on host: {self.hostname}, cmd: {cmd}, error: {error_output}", SSHExecutionWarning)
            return stdout.read().decode().strip()

    def close(self):
        with self.lock:
            if self.ssh.get_transport() and self.ssh.get_transport().is_active():
                self.execute_cmd(
                    """docker exec xdbcserver bash -c 'pids=$(pgrep xdbc-server); if [ "$pids" ]; then kill $pids; fi'""")
                self.execute_cmd(
                    """docker exec xdbcclient bash -c 'pids=$(pgrep xdbc-client); if [ "$pids" ]; then kill $pids; fi'""")
                self.ssh.close()


def create_ssh_connections(hosts, exp_num=0):
    """
    Create SSH connections for a list of hosts without username/password.

    Args:
        hosts (list): A list of hostnames or IP addresses.

    Returns:
        dict: A dictionary with hostnames as keys and SSH connections as values.
    """
    ssh_connections = {}
    i = 0
    for host in hosts:
        if i == exp_num:
            break
        try:
            ssh_connection = SSHConnection(host, "harry-ldap")
            ssh_connections[host] = ssh_connection
            i += 1
        except Exception as e:
            print(f"Failed to establish SSH connection to {host}: {str(e)}")

    return ssh_connections
