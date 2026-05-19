import paramiko
import getpass
import time
import os
from scp import SCPClient

#set hyperparameters
LR = 3e-5
EPOCHS = 10
BATCH_SIZE = 4#this is what UNER proposes when training on "all"
fineTuneMethod = "lora"

#set an experiment name (just derive it from the parameters) -> all stuff related to this training run will be saved here
experimentName = f"{fineTuneMethod}_lr{str(LR).split('.')[-1]}_E{EPOCHS}_B{BATCH_SIZE}"
outputDir = f"~/NER_proj/baseline_model_{experimentName}"

#login
username = input("Username: ")
password = getpass.getpass('Password: ')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('hpc.itu.dk', username=username, password=password)
print(f"Successfully logged in as {username}.")

targetLanguages = ["all the languages"]#just a placeholder so the code below doesn't break

lang2JobId = {}#map from target language to job id
for language in targetLanguages:
    #construct the command: (has to be in one go since each call to ssh.exec_command() produces a new session)
    command = (f"cd ~/NER_proj/hpc && "
                f"LEARNRATE={LR} "
                f"EPOCHS={EPOCHS} "
                f"BATCH_SIZE={BATCH_SIZE} "
                f"OUTPUT_DIR={outputDir} "
                f"FINETUNE={fineTuneMethod} "
                "sbatch baseline.job")
    # Submit the job
    stdin, stdout, stderr = ssh.exec_command(command)
    err = stderr.read().decode()
    if err:#if it's not the empty string, i.e. we got an error:
        print("STDERR:", err)
    print(stdout.read().decode())
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

ssh.close()