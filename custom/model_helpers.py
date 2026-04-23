import tqdm
import torch
import evaluate
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

# ------------------------------------------------------------

def train_model(model, dataloader: DataLoader, optimizer, epochs: int = 3, verbose: bool = True) -> None:
    """
    Function to train a model for the given amount of epochs.
    For each epoch, each batch in the given optimizer is processed once, updating model weights.
    
    Parameters
    ----------

    model
        The model to be trained.

    dataloader : torch.utils.data.DataLoader
        The class containing the batch-partitioned data.
    
    optimizer
        The optimizer to be used to train the model.
    
    epochs : int, optional (default= 3)
        The amount of epochs to train the model on.

    verbose : bool, optional (default= True)
        Flag variable to indicate wether or not the function should
        print status information while running.

    Returns
    -------

    None    
    """

    # move computation to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # set model up for training
    model.train()

    # perform epochs
    for epoch in range(epochs):

        # total loss over the current epoch
        total_loss = 0

        # use progressbar if verbose, else just iterate through the dataloader batches normally
        if verbose:
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Training Epoch {epoch+1}")
        else:
            pbar = dataloader

        # iterate through batches
        for batch in pbar:

            # zero model gradients
            optimizer.zero_grad()

            # move batch to available device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # obtain and store loss
            out = model(**batch)
            loss = out.loss
            total_loss += loss.item()
            
            # update weights
            loss.backward()
            optimizer.step()
        
        # print batch loss if verbose
        if verbose:
            print('Loss: ' + total_loss/len(pbar))

# ------------------------------------------------------------

def get_labels(predictions, references, idx_to_tag: dict[int, str]) -> tuple[list[list[str]], list[list[str]]]:
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

    # move labels to cpu, in numpy format
    predictions = predictions.cpu().numpy()
    references = references.cpu().numpy()
    
    # initialize arrays to store final labels
    true_predictions = []
    true_labels = []

    # translate labels from numerical to string format
    for i, example in enumerate(references):
      true_labels.append([idx_to_tag[idx] for idx in example if idx != -100])
      true_predictions.append([idx_to_tag[idx] for j, idx in enumerate(predictions[i,:]) if references[i, j] != -100])
    
    return true_predictions, true_labels

# ------------------------------------------------------------

def compute_metrics(preds, refs, metric: evaluate.EvaluationModule) -> dict[str, float]:
    """
    Based on given prediction - true label pairings,
    computes Precision, Recall, F1, and Accuracy measures.
    It returns them as a dictionary of name - value pairs.

    Arguments
    ---------
    
    preds :
        The array containing predicted labels.
    
    refs :
        The array containing reference / true labels.
    
    metric : evaluate.EvaluationModule
        The evaluation module needed to compute metrics.
    
    Returns
    -------

    metric_vals : dict[str, float]
        The dictionary containing name - value pair for the
        evaluation metrics (Precision, Recall, F1, Accuracy).
    """

    # compute metrics using the metric object
    results = metric.compute(predictions=preds, references=refs)
    return {
        "Precision": results["overall_precision"],
        "Recall": results["overall_recall"],
        "F1": results["overall_f1"],
        "Accuracy": results["overall_accuracy"],
    }

# ------------------------------------------------------------

def eval_model(model, dataloader: DataLoader, metric: evaluate.EvaluationModule, idx_to_tag: dict[int, str], verbose: bool = True):
    """
    Evaluates the given model on the data contained in the dataloader.
    The metrics are returned in a dictionary and, if verbose, printed.

    Parameters
    ----------

        model : 
            The model to be used to obtain predictions.

        dataloader : torch.utils.data.DataLoader
            The class containing the batch-partitioned data.

        metric : evaluate.EvaluationModule
            The evaluation module needed to compute metrics.
        
        idx_to_tag : dict[int, str]
            Dictionary of integer - string pairs to convert numerical labels
            into strinig labels, such as 'B-ORG', 'I-LOC', ...

        verbose : bool, optional (default= True)
            Flag variable to indicate wether or not the function should
            also print the metrics, other than returning.
            That is the case by default.
    
    Returns
    -------
         
    metric_vals : dict[str, float]
        The dictionary containing name - value pair for the
        evaluation metrics (Precision, Recall, F1, Accuracy).        
    """

    # move computation to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # set model up for evaluation
    model.eval()

    # use progressbar if verbose
    if verbose:
        validation_progress_bar = tqdm(range(len(dataloader)))
    
    # set up arrays to store predicted and true labels
    all_predictions = []
    all_labels = []

    # iterate through batches
    for step, batch in enumerate(dataloader):

        # move batch to available device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # obtain batch output
        with torch.no_grad():
            outputs = model(**batch)
        
        # softmax results to obtain predictions
        predictions = outputs.logits.argmax(dim=-1)
        
        # store results
        labels = batch["labels"]
        predicted_labels, true_labels = get_labels(predictions, labels, idx_to_tag)
        all_predictions.extend(predicted_labels)
        all_labels.extend(true_labels)

        # update progressbar if verbose
        if verbose:
            validation_progress_bar.update(1)

    # obtain validation metrics for the model
    validation_metrics = compute_metrics(all_predictions, all_labels, metric)
    
    # print results if verbose
    if verbose:
        print(validation_metrics)
    
    return validation_metrics

# ------------------------------------------------------------

def save_predictions(model, dataloader: DataLoader, tokenizer: PreTrainedTokenizerBase, filename: str, verbose: bool = True) -> None:
    """
    Stores predictions on the data of the given dataloader
    into a file.

    model : 
        The model to use to obtain predictions.

    dataloader : torch.utils.data.DataLoader
        The class containing the batch-partitioned data.
    
    tokenizer : PreTrainedTokenizerBase
        The tokenizer needed to decode sequences back to natural language.

    filename : str
        The name of the file in which to store predictions.

    verbose : bool, optional (default= True)
        Flag variable to indicate wether or not the function should
        print status information while running.

    Returns
    -------

    None
    """

    # move computation to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # set model up for evaluation
    model.eval()

    # setup progressbar if verbose 
    if verbose:
        validation_progress_bar = tqdm(range(len(dataloader)))
    
    # initialize arrays 
    all_predictions = []
    all_labels = []
    lines=[]

    # iterate through batches
    for step, batch in enumerate(dataloader):

        # iterate through sentences
        for sentence in batch["input_ids"]:

            # decode sentence, add empty line at the end
            lines.extend(tokenizer.decode(sentence, skip_special_tokens=True).split(" "))
            lines.append("")
        
        # move batch to available device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # obtain batch predictions
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)

        # store true and predicted labels
        labels = batch["labels"]
        predicted_labels, true_labels = get_labels(predictions, labels, dataloader.idx2tag)
        all_predictions.extend(predicted_labels)
        all_labels.extend(true_labels)
        
        if verbose:
            validation_progress_bar.update(1)

    # arrays to store labels for all sentences
    good_preds = []
    good_labels = []
    
    # iterate through sentences
    for i in range(len(all_predictions)):

        # append predicted and true labels to corresponding array
        for pred in all_predictions[i]:
            good_preds.append(pred)
        for label in all_labels[i]:
            good_labels.append(label)

    # write data to file -- !!! doesn't this misalign labels, since the labels paired with empty lines are ignored?
                        #       or do I just not know that the tokenizer and the model also predict a label for the empty line?
    with open(filename, "w") as f:
        for line, pred, true in zip(lines, good_preds, good_labels):
            if line != "":
                f.write(f"1\t{line}\t{pred}\t{true}\n")
            else:
                f.write("\n")