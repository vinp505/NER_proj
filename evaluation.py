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
metric = evaluate.load("seqeval")

if VERBOSE:
    print("Loading model ...")

# obtain all epoch folders from model folder
epochs = [e for e in os.listdir(MODEL_FOLDER) if e.startswith("epoch_")]
epochs.sort(key=lambda x: int(x.split("_")[1]))

# iterate through epochs, evaluate each one
for i, epoch in enumerate(epochs, 1):

    if VERBOSE:
        print(f"Evaluating: epoch {i}")
    
    # load lora weights for the epoch
    epoch_folder = os.path.join(MODEL_FOLDER, epoch)
    peft_model = PeftModel.from_pretrained(multi_model, epoch_folder)
    peft_model.to(device)
    peft_model.eval()  # inside eval_model() as well
    
    # iterate through languages, obtain metrics for each one
    for lang in language_data.lang_codes:

        if VERBOSE:
            print(f" > {lang}")
            
        test_set_l = test_dataset[lang]
        dataloader = DataLoader(test_set_l, batch_size= 8, collate_fn= multi_data_collator)
        custom.eval_model(peft_model, dataloader, metric, language_data.idx2tag, verbose= VERBOSE)

        # ADD NICER PRINTING AND SAVE RESULTS
    
    # delete model, empty cache before starting new run
    del peft_model
    torch.cuda.empty_cache()

