from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig
import torch
from torch.utils.data import DataLoader
import custom
import argparse
import evaluate

# ------------------------------------------------------------

# add a parser, include needed arguments
parser = argparse.ArgumentParser(description="Fine-tune baseline mBERT model")
parser.add_argument("-o", "--output", help="Output file for test set predictions", required=True)
parser.add_argument("-e", "--epochs", help="Number of fine-tuning epochs", required=True)
parser.add_argument("-l", "--learnRate", help="Learning Rate", required=True)
parser.add_argument("-b", "--batchSize", help="Batch Size", required=True)
parser.add_argument("-f", "--finetune", help="Fine Tuning Method, can be: <to be added>", required=True)
args = parser.parse_args()

# print main arguments
print("Output file: ", args.output, "Number of epochs: ", args.epochs)

# move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# parameter specification
MODEL_NAME = "google-bert/bert-base-multilingual-cased"
LR = 3e-5
EPOCHS = int(args.epochs)
BATCH_SIZE = 8

# load data and split sets
language_data = custom.LanguageData(MODEL_NAME)
data_splitter = custom.DataSplit(language_data)

# obtain needed datasets
train_dataset = data_splitter.get_train_set()
test_dataset = data_splitter.get_test_set("eng")

# load model and configuration
multi_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=7)
multi_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    config=multi_config
)
multi_model.to(device)

# load tokenizer, collator, and optimizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
multi_data_collator = DataCollatorForTokenClassification(tokenizer)
optimizer = torch.optim.AdamW(multi_model.parameters(), lr=LR)

# create dataloader objects to iterate through batches
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multi_data_collator)

metric = evaluate.load("seqeval")

# main part: train - eval
custom.train_model(multi_model, train_dataloader, optimizer, EPOCHS)
custom.eval_model(multi_model, test_dataloader, metric, language_data.idx2tag)
custom.save_predictions(multi_model, test_dataloader, tokenizer, args.output)
