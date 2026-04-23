import paramiko
import getpass
import time
import os
from scp import SCPClient

#the slurm log path has to match up with the script path specified in the script.job
slurm_log_path="~/NER_proj/hpc/baseline-10.out"

#set hyperparameters
outputFile = "~/NER_proj/final_predictions.iob2"#this is the location where the server saves the predictions -> we will get them from there and copy them to local
LR = 3e-5
EPOCHS = 20
BATCH_SIZE = 8
fineTuneMethod = "lora"
targetLanguages = ["eng", "slk", "dan", "rom", "chi"]#all languages that we wish to train models for
k = 10#number of training examples to include from the non-target languages

#login
username = input("Username:")
password = getpass.getpass('Password:')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('hpc.itu.dk', username=username, password=password)
print(f"Successfully logged in as {username}.")

# Upload the .job file -> such that edits are directly transferred without having to do a git pull on the hpc
local_job_file = "hpc/trainModel.job"
with SCPClient(ssh.get_transport()) as scp:
    scp.put(local_job_file, "~/NER_proj/hpc/trainModel.job")


lang2JobId = {}#map from target language to job id
for language in targetLanguages:
    #construct the command: (has to be in one go since each call to ssh.exec_command() produces a new session)
    command = (f"cd /home/{username}/NER_proj/hpc && "
                f"OUTPUT_FILE={outputFile} "
                f"LEARNRATE={LR} "
                f"EPOCHS={EPOCHS} "
                f"BATCH_SIZE={BATCH_SIZE} "
                f"FINETUNE={fineTuneMethod} "
                f"TARGET_LANG={language}"
                "sbatch trainModel.job")
    # Submit the job
    stdin, stdout, stderr = ssh.exec_command(command)
    job_id = int(stdout.read().decode().split()[-1])
    lang2JobId[language] = job_id

print("Submitted all jobs.")
print("Monitoring the situation closely ;)")

def get_job_state(ssh, job_id):
    _, stdout, _ = ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
    lines = stdout.read().decode().strip().split('\n')
    for line in lines:
        line = line.strip()
        if line:
            return line  # Return first non-empty state
    return "UNKNOWN"

#all the states in which a job is not running anymore
TERMINAL_STATES = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 
                   'NODE_FAIL', 'DEADLINE', 'BOOT_FAIL', 'OUT_OF_MEMORY'}

def all_done(states:dict) -> bool:
    return all([(state in TERMINAL_STATES) for state in states.values()])#only return true if all jobs are completed

while True:
    states = {lang: get_job_state(ssh, lang2JobId[lang]) for lang in targetLanguages}
    if all_done(states):
        print("All jobs terminated.")
        break
    #print status for all languages
    for lang in targetLanguages:
        print(f"{lang2JobId[lang]}({lang}):\tSTATUS: {states[lang]}")

print("End of Program.")

# #out = stdout.read().decode()
# err = stderr.read().decode()
# #print("STDOUT:", out)
# print("STDERR:", err)
# job_id = int(stdout.read().decode().split()[-1])
# print(f'Submitted job with ID {job_id}')

# #_______What follows is AI code



# def tail_remote_file(ssh, path, from_line=0):
#     """Read remote file from a given line offset, return new lines and updated offset."""
#     _, stdout, _ = ssh.exec_command(f'cat {path} 2>/dev/null')
#     all_lines = stdout.read().decode().splitlines()
#     new_lines = all_lines[from_line:]
#     return new_lines, len(all_lines)

# # Wait for job to start (log file won't exist while PENDING)
# print("Waiting for job to start...")
# while True:
#     state = get_job_state(ssh, job_id)
#     if 'RUNNING' in state:
#         print(f"Job {job_id} is now running. Streaming output...\n")
#         break
#     elif any(s in state for s in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']):
#         print(f"Job ended before we could stream: {state}")
#         break
#     else:
#         print(f"State: {state} — waiting...")
#         time.sleep(10)

# # Stream the log live
# lines_seen = 0
# job_finished = False

# while not job_finished:
#     # Print any new lines from the log
#     new_lines, lines_seen = tail_remote_file(ssh, slurm_log_path, from_line=lines_seen)
#     for line in new_lines:
#         print(line)

#     # Check job state
#     state = get_job_state(ssh, job_id)
#     if any(s in state for s in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']):
#         # One final read to catch any last output
#         new_lines, lines_seen = tail_remote_file(ssh, slurm_log_path, from_line=lines_seen)
#         for line in new_lines:
#             print(line)
#         print(f'\nJob {job_id} finished with state: {state}')
#         job_finished = True
#     else:
#         time.sleep(10)

# # Download final outputs
# current_directory = os.getcwd()
# slurm_out = f"slurm-{job_id}.out"
# with SCPClient(ssh.get_transport()) as scp:
#     scp.get(slurm_log_path, os.path.join(current_directory, slurm_out))
# print(f"Downloaded Slurm log to {slurm_out}")

# local_output_filename = "final_predictions.iob2"
# with SCPClient(ssh.get_transport()) as scp:
#     scp.get(outputFile, local_output_filename)
# print(f'Downloaded output file to {local_output_filename}')

ssh.close()