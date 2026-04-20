import paramiko
import getpass
import time
import os
from scp import SCPClient

#set hyperparameters
outputFile = "~/NER_proj/final_predictions.iob2"
LR = 3e-5
EPOCHS = 20
BATCH_SIZE = 8
fineTuneMethod = "full"

#login
username = input("Username:")
password = getpass.getpass('Password:')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('hpc.itu.dk', username=username, password=password)
print(f"Successfully logged in as {username}.")

# Upload the .job file -> such that edits are directly transferred without having to do a pull on the hpc
local_job_file = "hpc/baseline.job"
with SCPClient(ssh.get_transport()) as scp:
    scp.put(local_job_file, "~/NER_proj/hpc/baseline.job")

#construct the command: (has to be in one go since each call to ssh.exec_command() produces a new session)
command = (f"cd /home/{username}/NER_proj/hpc && "
            f"OUTPUT_FILE={outputFile} "
            f"LEARNRATE={LR} "
            f"EPOCHS={EPOCHS} "
            f"BATCH_SIZE={BATCH_SIZE} "
            f"FINETUNE={fineTuneMethod} "
            "sbatch baseline.job")

# Submit the job
stdin, stdout, stderr = ssh.exec_command(command)
#out = stdout.read().decode()
err = stderr.read().decode()
#print("STDOUT:", out)
print("STDERR:", err)
job_id = int(stdout.read().decode().split()[-1])
print(f'Submitted job with ID {job_id}')

job_finished = False
while not job_finished:
    time.sleep(10)
    stdin, stdout, stderr = ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
    output = stdout.read().decode().strip().split('\n')
    for line in output:
        if 'COMPLETED' in line:
            job_finished = True
            print(f'Job {job_id} has completed')
            break
        elif 'RUNNING' in line:
            print(f'Job {job_id} is still running...')
            break
        elif 'PENDING' in line:
            print(f'Pending...')

current_directory = os.getcwd()
#get the slurm log:
slurm_out = f"baseline-10.out"
with SCPClient(ssh.get_transport()) as scp:
    scp.get(slurm_out, os.path.join(current_directory, slurm_out))
print("SLURM_OUTPUT:")
print(slurm_out)
print("________________________---")

local_output_filename = os.path.join(current_directory, outputFile)

with SCPClient(ssh.get_transport()) as scp:
    scp.get(outputFile, local_output_filename)
print(f'Downloaded .out file to {local_output_filename}')

# with open(local_output_filename) as file:
#     for line in file:
#         print(line)

ssh.close()