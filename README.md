# Named Entity Recognition — Cross-Lingual Fine-Tuning

A research project exploring cross-lingual transfer and forgetting for language-specific finetuning from a shared baseline in the task of Named Entity Recognition (NER).

Dataset: Universal NER
[Github](https://github.com/universal-ner/universal-ner)

---

## Reproducing Results

Follow these steps in order. Steps 3–5 require your HPC credentials; step 6 additionally requires a Hugging Face token (read-only access is sufficient).

### 1. Clone the Repository

Clone to the home directory of your user on the HPC:

```bash
git clone https://github.com/vinp505/NER_proj.git ~/NER_proj
```
Run this command also on your local machine.

### 2. Set Up the Environment

```bash
cd ~/NER_proj/hpc
sbatch setup.job
```
You might want to setup the environment locally as well, which you can do by pip-installing the requirements.txt (for running the queue scripts)

### 3. Cache the Base Model *(optional, but recommended)*

```bash
sbatch downloadModel.job
```

### 4. Train the Baseline Model

```bash
cd ~/NER_proj
python queueBaseline.py
```
Run the above command locally, it will queue the necessary job on the hpc.

### 5. Train Language-Specific Fine-Tunes

Trains five models starting from the baseline weights, one per language (`eng`, `slk`, `dan`, `rom`, `chi`):

```bash
python queueTraining.py
```
Run the above command locally, it will queue the necessary job on the hpc.

### 6. Evaluate All Models

Evaluates each of the five models across all test sets. Results are saved to the `evaluation_results/` directory. Provide a read-only Hugging-Face token:

```bash
python queueEval.py
```
Run the above command locally, it will queue the necessary job on the hpc.

### 7. Copy Evaluation Results to Your Local Machine

```bash
scp -r <username>@hpc.itu.dk:/home/<username>/NER_proj/evaluation_results <path to the repository on your local machine>
```

### 8. Merge Evaluation Results

Open and run all cells in the notebook:

```
evaluation_results/merging_csv.ipynb
```

This produces `evaluation_results/evaluation_ftmodel_eng.csv`.

### 9. Visualise Results

1. Copy the merged CSV to the visualization directory:
   ```bash
   cp evaluation_results/evaluation_ftmodel_eng.csv visualization/
   ```
2. Open the Tableau workbook inside the `visualization/` directory — it will display the plots included in the paper.

---

## Requirements

- Access to ITU's hpc
- A [Hugging Face](https://huggingface.co) account with a read-only API token (for the evaluation part)
- Tableau Desktop (for visualisations)
