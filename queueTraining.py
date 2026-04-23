import paramiko
import getpass
import time
import os
from scp import SCPClient

#set hyperparameters
outputDir = "~/NER_proj/final_predictions.iob2"#this is the location where the server saves the predictions -> we will get them from there and copy them to local
LR = 3e-5
EPOCHS = 20
BATCH_SIZE = 8
fineTuneMethod = "lora"
targetLanguages = ["eng", "slk", "dan", "rom", "chi"]#all languages that we wish to train models for
k = 10#number of training examples to include from the non-target languages

#set an experiment name (just derive it from the parameters) -> all stuff related to this training run will be saved here
experimentName = f"{fineTuneMethod}_lr{round(LR,3)}_E{EPOCHS}_B{BATCH_SIZE}_k{k}"
outputDir = f"~NER_proj/{experimentName}"

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
    command = (f"cd ~/NER_proj/hpc && "
                f"OUTPUT_DIR={outputDir} "
                f"LEARNRATE={LR} "
                f"EPOCHS={EPOCHS} "
                f"BATCH_SIZE={BATCH_SIZE} "
                f"FINETUNE={fineTuneMethod} "
                f"K_NON_TARGET={k}"
                f"TARGET_LANG={language} "
                "sbatch trainModel.job")
    # Submit the job
    stdin, stdout, stderr = ssh.exec_command(command)
    err = stderr.read().decode()
    if err:#if it's not the empty string, i.e. we got an error:
        print("STDERR:", err)
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
    print("______________________________________")

print("End of Program.")

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