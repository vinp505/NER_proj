"""Script to carry out fine-tuning of a RoBERTa model on equal amount of senteces for each language to obtain the baseline model."""

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
parser = argparse.ArgumentParser(description="Fine-tune baseline model")
parser.add_argument("-e", "--epochs", help="Number of fine-tuning epochs", required=False)
parser.add_argument("-lr", "--learnRate", help="Learning Rate", required=False)
parser.add_argument("-b", "--batchSize", help="Batch Size", required=False)
parser.add_argument("-f", "--finetune", help="Fine Tuning Method, can be: <to be added>", required=False)
parser.add_argument("-v", "--verbose", help= "Boolean flag to indicate whether or not the script should periodically print status updates", required= False)
parser.add_argument("-o", "--output", help="Output folder for trained model", required=True)
args = parser.parse_args()

# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# parameter specification
MODEL_NAME = "FacebookAI/xlm-roberta-base"
OUTPUT_DIR = pathlib.Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EPOCHS = int(args.epochs) if args.epochs != None else 20
LR = float(args.learnRate) if args.learnRate != None else 0.0001
BATCH_SIZE = int(args.batchSize) if args.batchSize != None else 64
ACCUMUL_STEPS = BATCH_SIZE // 8
FINETUNE_METHOD = args.finetune if args.finetune != None else "lora"
VERBOSE = bool(args.verbose) if args.verbose != None else False

# print main arguments
print("Model folder: ", OUTPUT_DIR, "\nNumber of epochs: ", EPOCHS)

# ------------------------------------------------------------

if VERBOSE:
    print("Loading data ...")

# load data and split sets
language_data = custom.LanguageData(MODEL_NAME, verbose= VERBOSE)
data_splitter = custom.DataSplit(language_data, target_lang= "all", verbose= VERBOSE)

# obtain needed dataset -> balanced across languages
train_dataset = data_splitter.get_train_set()

if VERBOSE:
    print("Loading base model ...")

# load model and configuration -> oob model + baseline lora weights
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
    
    # configure lora finetuning
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=8,
        lora_alpha=32,
        init_lora_weights="gaussian",
        target_modules=["query", "key", "value", "dense"]
    )
    peft_model = get_peft_model(multi_model, peft_config)
    peft_model.print_trainable_parameters()
    peft_model.to(device)

    if VERBOSE:
        print("Setting up baseline model finetuning ...")
        
    # load tokenizer, collator, and optimizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    multi_data_collator = DataCollatorForTokenClassification(tokenizer)

    # create dataloader objects to iterate through batches
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_data_collator)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
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
        print(f"Finetuning baseline model ...")

    # train and save the model
    trainer.train()
    peft_model.save_pretrained(OUTPUT_DIR)
    
    # rename checkpoint files to be more human readable
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))

    for i, checkpoint in enumerate(checkpoints, 1):
        old_path = os.path.join(OUTPUT_DIR, checkpoint)
        new_path = os.path.join(OUTPUT_DIR, f"epoch_{i}")
        os.rename(old_path, new_path)

    print("Baseline model finetuned and saved.")

else:
    print("For now, the only supported finetune method is \"lora\".")