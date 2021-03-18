import paramiko
from src.utils.proxy_utils import load_user_password, percentage_encoding_to_regular


def run_command(ssh_ptr, command, print_flag=False):
    ssh_stdin, ssh_stdout, ssh_stderr = ssh_ptr.exec_command(command)
    stdin = "stdin={}".format("")
    stdout = "stdout={}".format(str(ssh_stdout.read().decode("utf-8")))
    stderr = "stderr={}".format(str(ssh_stderr.read().decode("utf-8")))
    return stdin, stdout, stderr


def get_nvidia_status():
    user, password = load_user_password()
    passwork_not_percentage = percentage_encoding_to_regular(password)
    # print(f"user={user}; password={password}; regular={passwork_not_percentage}")
    ip = "10.64.66.32"

    head_ssh = paramiko.SSHClient()
    head_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    head_ssh.connect(ip, username=user, password=passwork_not_percentage, allow_agent=True, look_for_keys=True)

    stdin, stdout, stderr = run_command(head_ssh, " nvidia-smi stats -i 0 -c 1 |grep gpuUtil")
    print(stdout)

    head_ssh.close()


if __name__ == "__main__":
    get_nvidia_status()
