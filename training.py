"""Script to carry out language-specific fine-tuning of the baseline RoBERTa model (previously fine-tuned on equal amount of senteces for each language)."""

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

# don't redownload the models
cachePath = os.path.expanduser("~/NER_proj/hf_cache")
os.environ["HF_HOME"] = cachePath
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # prevents any attempt to re-download

# add a parser, include needed arguments
parser = argparse.ArgumentParser(description="Fine-tune model")
parser.add_argument("-l", "--language", help="Fine tuning target language code: 'eng', 'slk', 'dan', 'rom', 'chi', or 'all' for all languages", required=True)
parser.add_argument("-o", "--output", help="Output folder for trained model", required=False)
parser.add_argument("-e", "--epochs", help="Number of fine-tuning epochs", required=False)
parser.add_argument("-lr", "--learnRate", help="Learning Rate", required=False)
parser.add_argument("-b", "--batchSize", help="Batch Size", required=False)
parser.add_argument("-f", "--finetune", help="Fine Tuning Method, can be: <to be added>", required=False)
parser.add_argument("-k", "--kNonTarget", help="Number of training examples to include from the non-target languages", required=False)
parser.add_argument("-v", "--verbose", help= "Boolean flag to indicate whether or not the script should periodically print status updates", required= False)
parser.add_argument("-bd", "--baselineDir", help="Directory where the script will look for the baseline (this is the starting point for training).")
args = parser.parse_args()

# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# parameter specification
MODEL_NAME = "FacebookAI/xlm-roberta-base"
BASELINE_FOLDER = pathlib.Path(args.baselineDir) if args.baselineDir != None else "baseline_model"
TARGET_LANG = args.language
OUTPUT_DIR = pathlib.Path(args.output) if args.output != None else pathlib.Path(f"finetuned_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_PATH = OUTPUT_DIR / (f"finetuned_model_{TARGET_LANG}")
EPOCHS = int(args.epochs) if args.epochs != None else 20
LR = float(args.learnRate) if args.learnRate != None else 1e-4
BATCH_SIZE = int(args.batchSize) if args.batchSize != None else 64
ACCUMUL_STEPS = BATCH_SIZE // 8 if BATCH_SIZE >= 8 else 1
FINETUNE_METHOD = args.finetune if args.finetune != None else "lora"
K = int(args.kNonTarget) if args.kNonTarget != None else 10
VERBOSE = args.verbose if args.verbose != None else False

# print main arguments
print("Output folder: ", OUTPUT_DIR, "\nModel folder: ", MODEL_OUTPUT_PATH, "\nNumber of epochs: ", EPOCHS)

# ------------------------------------------------------------

if VERBOSE:
    print("Loading data ...")

# load data and split sets
language_data = custom.LanguageData(MODEL_NAME, verbose= VERBOSE)
data_splitter = custom.DataSplit(language_data, target_lang=TARGET_LANG, k=K, verbose= VERBOSE)

# obtain needed datasets
train_dataset = data_splitter.get_train_set() #assembles correct training set for the requested target language(s)

if VERBOSE:
    print("Loading base model ...")

# load model and configuration -> oob model
multi_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=7)
multi_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    config=multi_config
)
# ------------------------------------------------------------

#____ADD SOME IF STATEMENT HERE TO CHECK IF THE METHOD IS LORA or FULL FINETUNE etc...
if FINETUNE_METHOD.lower() == "lora":

    if VERBOSE:
        print("Setting up LoRA finetuning ...")
    
    # obtain baseline model (lora)
    peft_model = PeftModel.from_pretrained(
        multi_model, 
        BASELINE_FOLDER, 
        is_trainable=True
    )
    #peft_model.base_model.trainable = False
    peft_model.print_trainable_parameters()
    peft_model.to(device)

    if VERBOSE:
        print("Setting up model finetuning ...")

    # load tokenizer, collator, and optimizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    multi_data_collator = DataCollatorForTokenClassification(tokenizer)

    # create dataloader objects to iterate through batches
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_data_collator)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE//ACCUMUL_STEPS,
        gradient_accumulation_steps=ACCUMUL_STEPS,
        num_train_epochs=EPOCHS,
        save_strategy="epoch",
        eval_strategy="no",
        do_eval= False,
        logging_strategy="epoch",
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=multi_data_collator
    )

    if VERBOSE:
        print(f"Finetuning model ...")
        
    # train and save the model
    trainer.train()
    peft_model.save_pretrained(MODEL_OUTPUT_PATH)
    
    # rename checkpoint files to be more human readable
    checkpoints = [d for d in os.listdir(MODEL_OUTPUT_PATH) if d.startswith("checkpoint-")]
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))

    for i, checkpoint in enumerate(checkpoints, 1):
        old_path = os.path.join(MODEL_OUTPUT_PATH, checkpoint)
        new_path = os.path.join(MODEL_OUTPUT_PATH, f"epoch_{i}")
        os.rename(old_path, new_path)

    print("Model finetuned and saved.")

else:
    print("For now, the only supported finetune method is \"lora\".")