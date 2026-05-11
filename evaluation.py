"""Script to carry out evaluation of a language-specific fine-tuned version of the baseline RoBERTa model (previously fine-tuned on equal amount of senteces for each language)."""

# ------------------------------------------------------------

from tqdm.auto import tqdm
from functools import partial
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
import custom
import argparse
import evaluate
import pathlib
import os
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# ------------------------------------------------------------

# add a parser, include needed arguments
parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("-l", "--language", help="Target language code of the model to be evaluated: 'all' (baseline model), 'eng', 'slk', 'dan', 'rom', 'chi'", required=True)
parser.add_argument("-m", "--modelFolder", help="Folder containing the model (folder) to be evaluated", required=False)
parser.add_argument("-d", "--evalDirectory", help="Directory in which to save the evaluation results", required=False)
parser.add_argument("-v", "--verbose", help= "Boolean flag to indicate whether or not the script should periodically print status updates", required= False)
args = parser.parse_args()

# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# parameter specification
MODEL_NAME = "FacebookAI/xlm-roberta-base"
TARGET_LANG = args.language

if TARGET_LANG == 'all':
    MODEL_FOLDER = "baseline_model"
else:
    MODEL_DIR = pathlib.Path(args.modelFolder) if args.modelFolder != None else pathlib.Path(f"finetuned_models")
    MODEL_FOLDER = MODEL_DIR / (f"finetuned_model_{TARGET_LANG}")

EVAL_DIR = pathlib.Path(args.evalDirectory) if args.evalDirectory != None else pathlib.Path("evaluation_results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
VERBOSE = bool(args.verbose) if args.verbose != None else False

# print main arguments
print("Model folder: ", MODEL_FOLDER)
print(f"Target language: {TARGET_LANG}")

# ------------------------------------------------------------

if VERBOSE:
    print("Loading data ...")

# load data and split sets
language_data = custom.LanguageData(MODEL_NAME, verbose= VERBOSE)
data_splitter = custom.DataSplit(language_data, verbose= VERBOSE)

# obtain needed datasets
test_dataset = data_splitter.get_test_set()

# load model and configuration -> oob model
multi_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=7)
multi_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    config=multi_config
)

# load tokenizer, collator, and eval module
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
multi_data_collator = DataCollatorForTokenClassification(tokenizer)
eval_module = evaluate.load("seqeval")

if VERBOSE:
    print("Loading model ...")

# obtain all epoch folders from model folder
epochs = [e for e in os.listdir(MODEL_FOLDER) if e.startswith("epoch_")]
epochs.sort(key=lambda x: int(x.split("_")[1]))


# initialize dictionaries to store results

lang_eval_data = {}
lang_f1_records = {}

for lang in language_data.lang_codes:

    lang_eval_data[lang] = {
        "F1" : [],
        "Precision" : [],
        "Recall" : [],
        "Accuracy" : []
    }
    lang_f1_records[lang] = {
        "High" : {"F1" : -0.1, "Epoch" : None},
        "Low" : {"F1" : 1.1, "Epoch" : None}
    }

# iterate through epochs, evaluate each one
for i, epoch in enumerate(epochs, 1):

    if VERBOSE:
        print(f"Evaluating: epoch {i}")
    
    # load model
    multi_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=7)
    multi_model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        config=multi_config
    )
    
    # load lora weights for the epoch
    epoch_folder = os.path.join(MODEL_FOLDER, epoch)
    peft_model = PeftModel.from_pretrained(multi_model, epoch_folder)
    peft_model.to(device)
    peft_model.eval()  # inside eval_model() as well
    
    # iterate through languages, obtain metrics for each one
    for lang in language_data.lang_codes:

        if VERBOSE:
            print(f" > {lang}")

        # obtain metrics
        test_set_l = test_dataset[lang]
        dataloader = DataLoader(test_set_l, batch_size= 64, collate_fn= multi_data_collator)
        metrics = custom.eval_model(peft_model, dataloader, eval_module, language_data.idx2tag, verbose= VERBOSE)
        

        # store metrics

        for k, v in metrics.items():
            lang_eval_data[lang][k].append(v)
        
        if metrics['F1'] < lang_f1_records[lang]["Low"]["F1"]:
            lang_f1_records[lang]["Low"]["F1"] = metrics['F1']
            lang_f1_records[lang]["Low"]["Epoch"] = i
        
        if metrics['F1'] > lang_f1_records[lang]["High"]["F1"]:
            lang_f1_records[lang]["High"]["F1"] = metrics['F1']
            lang_f1_records[lang]["High"]["Epoch"] = i
            
    
    # delete model, empty cache before starting new run
    del multi_model
    del peft_model
    torch.cuda.empty_cache()

# print results
print("\n\n")
for lang, data in lang_eval_data.items():
    print("\n+" + "-"*63 + "+")
    print(f"|{('Language: ' + lang):^63}|")
    print("+" + "-"*63 + "+")
    print(f"|{f'Highest F1: {lang_f1_records[lang]['High']['F1']:.3f} (Epoch {lang_f1_records[lang]['High']['Epoch']})  |  Lowest F1: {lang_f1_records[lang]['Low']['F1']:.3f} (Epoch {lang_f1_records[lang]['Low']['Epoch']})':^63}|")
    print("+" + "-"*63 + "+")
    print(f"|{'Epoch':^11}|{'F1':^12}|{'Precision':^12}|{'Recall':^12}|{'Accuracy':^12}|")
    print("+" + "-"*11 + ('+' + "-"*12)*4 + '+')

    for i in range(len(epochs)):
        print(f"|{(i+1):^11}|{data['F1'][i]:>12.3f}|{data['Precision'][i]:>12.3f}|{data['Recall'][i]:>12.3f}|{data['Accuracy'][i]:>12.3f}|")
    print("+" + "-"*63 + "+")

# save results in csv format
with open(str(EVAL_DIR) + f'/evaluation_ftmodel_{TARGET_LANG}.csv', 'w') as f:
    
    f.write("lang,epoch,F1,precision,recall,accuracy\n")

    for lang, data in lang_eval_data.items():
        for i in range(len(epochs)):
            f.write(f"{lang},{(i+1)},{data['F1'][i]:.3f},{data['Precision'][i]:.3f},{data['Recall'][i]:.3f},{data['Accuracy'][i]:.3f}\n")
