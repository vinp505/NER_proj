from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoConfig
import torch
from torch.utils.data import DataLoader
from custom import LanguageData, DataSplit
import argparse
import evaluate

parser = argparse.ArgumentParser(description="Fine-tune baseline mBERT model")

parser.add_argument("-o", "--output", help="Output file for test set predictions", required=True)

parser.add_argument("-e", "--epochs", help="Number of fine-tuning epochs", required=True)

parser.add_argument("-l", "--learnRate", help="Learning Rate", required=True)

parser.add_argument("-b", "--batchSize", help="Batch Size", required=True)

parser.add_argument("-f", "--finetune", help="Fine Tuning Method, can be: <to be added>", required=True)

args = parser.parse_args()

print(args.output, args.epochs)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
LR = 3e-5
EPOCHS = int(args.epochs)
BATCH_SIZE = 8


language_data = LanguageData(MODEL_NAME)

data_splitter = DataSplit(language_data)
train_dataset = data_splitter.get_train_set()
test_dataset = data_splitter.get_test_set("eng")

multi_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=7)
multi_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    config=multi_config
)
multi_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
multi_data_collator = DataCollatorForTokenClassification(tokenizer)
optimizer = torch.optim.AdamW(multi_model.parameters(), lr=LR)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multi_data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multi_data_collator)

idx_to_tag = language_data.idx2tag

def train_model(model, dataloader, optimizer, epochs, verbose):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {epoch+1}")
        for step, batch in pbar:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        if verbose:
            print('Loss: ' + total_loss/len(pbar))

metric = evaluate.load("seqeval")

def get_labels(predictions, references):
    """
    Convert model outputs (logits) and references (label IDs)
    into human-readable label names, ignoring subword tokens.

    Args:
        predictions: A PyTorch tensor with shape [batch_size, seq_length],
                     containing the predicted label IDs for each token.
        references:  A PyTorch tensor with the true label IDs for each token.

    Returns:
        true_predictions, true_labels:
        - Each is a list of lists of strings.
        - Outer list = batch dimension
        - Inner list = predicted or true labels for each token in that example
        - We skip any token whose label == -100 (these are subword tokens or padding).

    Example:
        Suppose label_list = ["O", "B-PER", "I-PER"],
        predictions = [[0, 1, 2], [0, 0, 1]],
        references  = [[0, 1, 2], [0, 0, -100]]

        Then,
        true_predictions might be [["O", "B-PER", "I-PER"], ["O", "O"]]
        true_labels      might be [["O", "B-PER", "I-PER"], ["O", "O"]]
    """
    predictions = predictions.cpu().numpy()
    references = references.cpu().numpy()
    true_predictions = []
    true_labels = []
    for i, example in enumerate(references):
      true_labels.append([idx_to_tag[idx] for idx in example if idx != -100])
      true_predictions.append([idx_to_tag[idx] for j, idx in enumerate(predictions[i,:]) if references[i, j] != -100])
    return true_predictions, true_labels

def compute_metrics(preds, refs):
    results = metric.compute(predictions=preds, references=refs)
    return {
        "Precision": results["overall_precision"],
        "Recall": results["overall_recall"],
        "F1": results["overall_f1"],
        "Accuracy": results["overall_accuracy"],
    }

def eval_model(model, dataloader):
    model.eval()
    validation_progress_bar = tqdm(range(len(dataloader)))
    all_predictions = []
    all_labels = []
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predicted_labels, true_labels = get_labels(predictions, labels)
        all_predictions.extend(predicted_labels)
        all_labels.extend(true_labels)
        validation_progress_bar.update(1)

    validation_metrics = compute_metrics(all_predictions, all_labels)
    print(validation_metrics)

def save_predictions(model, dataloader, tokenizer, filename):
    model.eval()
    validation_progress_bar = tqdm(range(len(dataloader)))
    all_predictions = []
    all_labels = []
    lines=[]
    for step, batch in enumerate(dataloader):
        for sentence in batch["input_ids"]:
            lines.extend(tokenizer.decode(sentence, skip_special_tokens=True).split(" "))
            lines.append("")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predicted_labels, true_labels = get_labels(predictions, labels)
        all_predictions.extend(predicted_labels)
        all_labels.extend(true_labels)
        validation_progress_bar.update(1)

    good_preds = []
    good_labels = []
    
    for i in range(len(all_predictions)):
        for pred in all_predictions[i]:
            good_preds.append(pred)
        for label in all_labels[i]:
            good_labels.append(label)

    with open(filename, "w") as f:
        for line, pred, true in zip(lines, good_preds, good_labels):
            if line != "":
                f.write(f"1\t{line}\t{pred}\t{true}\n")
            else:
                f.write("\n")

train_model(multi_model, train_dataloader, optimizer, EPOCHS)
eval_model(multi_model, test_dataloader)
save_predictions(multi_model, test_dataloader, tokenizer, args.output)
