import paramiko
import getpass
import time
import os
from scp import SCPClient

#login
username = input("Username:")
password = getpass.getpass('Password:')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('hpc.itu.dk', username=username, password=password)
print(f"Successfully logged in as {username}.")

job_id = int(input("Please enter job id:"))
slurm_log_path=f"~/NER_proj/hpc/modelTrain-{job_id}.out"#this has to match up with the path specified in the trainModel.job

def get_job_state(ssh, job_id):
    _, stdout, _ = ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
    lines = stdout.read().decode().strip().split('\n')
    for line in lines:
        line = line.strip()
        if line:
            return line  # Return first non-empty state
    return "UNKNOWN"


def tail_remote_file(ssh, path, from_line=0):
    """Read remote file from a given line offset, return new lines and updated offset."""
    _, stdout, _ = ssh.exec_command(f'cat {path} 2>/dev/null')
    all_lines = stdout.read().decode().splitlines()
    new_lines = all_lines[from_line:]
    return new_lines, len(all_lines)

# Wait for job to start (log file won't exist while PENDING)
print("Waiting for job to start...")
while True:
    state = get_job_state(ssh, job_id)
    if 'RUNNING' in state:
        print(f"Job {job_id} is now running. Streaming output...\n")
        break
    elif any(s in state for s in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']):
        print(f"Job ended before we could stream: {state}")
        break
    else:
        print(f"State: {state} — waiting...")
        time.sleep(10)

# Stream the log live
lines_seen = 0
job_finished = False

while not job_finished:
    # Print any new lines from the log
    new_lines, lines_seen = tail_remote_file(ssh, slurm_log_path, from_line=lines_seen)
    for line in new_lines:
        print(line)

    # Check job state
    state = get_job_state(ssh, job_id)
    if any(s in state for s in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']):
        # One final read to catch any last output
        new_lines, lines_seen = tail_remote_file(ssh, slurm_log_path, from_line=lines_seen)
        for line in new_lines:
            print(line)
        print(f'\nJob {job_id} finished with state: {state}')
        job_finished = True
    else:
        time.sleep(10)

ssh.close()