from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig, TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader
import custom
import argparse
import evaluate
import pathlib
from peft import LoraConfig, TaskType, get_peft_model

# ------------------------------------------------------------

# add a parser, include needed arguments
parser = argparse.ArgumentParser(description="Fine-tune baseline mBERT model")
parser.add_argument("-l", "--language", help="Fine tuning target language code, e.g., 'de' for German, 'all' for all languages", required=True)
parser.add_argument("-o", "--output", help="Output folder for trained model and evaluation results", required=True)
parser.add_argument("-e", "--epochs", help="Number of fine-tuning epochs", required=True)
parser.add_argument("-lr", "--learnRate", help="Learning Rate", required=True)
parser.add_argument("-b", "--batchSize", help="Batch Size", required=True)
parser.add_argument("-f", "--finetune", help="Fine Tuning Method, can be: <to be added>", required=True)
parser.add_argeument("-k", "-kNonTarget", help="Number of training examples to include from the non-target languages.", required=True)
args = parser.parse_args()

# print main arguments
print("Output folder: ", args.output, "Number of epochs: ", args.epochs)

# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# parameter specification
MODEL_NAME = "FacebookAI/xlm-roberta-base"
LR = float(args.learnRate)
EPOCHS = int(args.epochs)
BATCH_SIZE = int(args.batchSize)
OUTPUT_DIR = pathlib.Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_PATH = OUTPUT_DIR / MODEL_NAME.split("/")[-1] + "_finetuned_" + args.language

# load data and split sets
language_data = custom.LanguageData(MODEL_NAME)
data_splitter = custom.DataSplit(language_data, target_lang=args.language)

# obtain needed datasets
train_dataset = data_splitter.get_train_set()
test_dataset = data_splitter.get_test_set("eng")

# load model and configuration
multi_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=7)
multi_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    config=multi_config
)

#____ADD SOME IF STATEMENT HERE TO CHECK IF THE METHOD IS LORA or FULL FINETUNE etc...

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

# load tokenizer, collator, and optimizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
multi_data_collator = DataCollatorForTokenClassification(tokenizer)

# create dataloader objects to iterate through batches
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_data_collator)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=multi_data_collator,
    compute_metrics=custom.compute_metrics,
)

trainer.train()

peft_model.save_pretrained(OUTPUT_DIR)