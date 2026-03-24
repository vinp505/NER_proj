import requests
import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Literal

class LanguageData():
    """
    Class to load and store Dataset objects for each language.
    Given the language code, its associated entry in the lang2data dictionary is
    a list of Dataset objects.
    For standard language data, this list consists of, in order, a train, a dev and a test set.
    Romanian and German are exceptions, with only one dataset each.
    Information about the smallest data sets is also stored.

    The language codes are associated with the corresponding language in the following tuples: 
    ("slk", Slovakian), ("eng", English), ("swe", Swedish), ("nor", Norwegian), ("heb", Hebrew), ("rom", Romanian),
    ("por", Portuguese), ("ger", German), ("chi", Chinese), ("hrv", Croatian), ("srb", Serbian), ("dan" Danish).

    Parameters
    -----------

    model: str
        The string containing the base pretrained model name,
        needed to retrieve the tokenizer.
    
    verbose: bool, optional (default= False)
        Flag to signal wether or not the class should print
        information about the data loading process.
    
    Methods
    --------

    **get_lang_data(lang)**
        Retrieve the list containing Dataset objects for the specified language.
    
    **get_smallest_set_size(set)**
        Retrieve the size of the smallest set of the specified type, together with its language code.
        It is possible to ask for the train, dev, or test sets, 
        as well as all of them, in which case a dictionary will be returned.
    """

    # model to obtain tokenizer
    def __init__(self, model: str, verbose: bool = False):

        # associate each language string code to partial data link
        lang2link = {
            "slk" : "https://raw.githubusercontent.com/UniversalNER/UNER_Slovak-SNK/main/sk_snk-ud",
            "eng" : "https://raw.githubusercontent.com/UniversalNER/UNER_English-EWT/master/en_ewt-ud",
            "swe" : "https://raw.githubusercontent.com/UniversalNER/UNER_Swedish-Lines/master/sv_lines-ud",
            "nor" : "https://raw.githubusercontent.com/UniversalNER/UNER_Norwegian-NDT/master/nno_norne-ud",
            "heb" : "https://raw.githubusercontent.com/UniversalNER/UNER_Hebrew-HTB/master/he_htb-ud",
            "rom" : "https://raw.githubusercontent.com/UniversalNER/UNER_Romanian-LegalNERo/master/ro_legalnero.iob2",  # hard-coded bc UNER is inconsistent
            "por" : "https://raw.githubusercontent.com/UniversalNER/UNER_Portuguese-Bosque/master/pt_bosque-ud",
            "ger" : "https://raw.githubusercontent.com/UniversalNER/UNER_German-PUD/master/de_pud-ud",
            "chi" : "https://raw.githubusercontent.com/UniversalNER/UNER_Chinese-GSD/master/zh_gsd-ud",
            "hrv" : "https://raw.githubusercontent.com/UniversalNER/UNER_Croatian-SET/main/hr_set-ud",
            "srb" : "https://raw.githubusercontent.com/UniversalNER/UNER_Serbian-SET/main/sr_set-ud",
            "dan" : "https://raw.githubusercontent.com/UniversalNER/UNER_Danish-DDT/main/da_ddt-ud"
        }

        # store model and entity tags conversion dictionaries
        self.model = model
        self.tag2idx = {
            "O": 0, 
            'B-ORG': 1, 'I-ORG': 2, 'B-OTH': 1, 'I-OTH': 2,  # for an out-of-insturctions tag 'OTHER', convert to PERSON
            'I-PER': 3, 'B-PER': 4, 
            'B-LOC': 5, 'I-LOC': 6}
        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}

        # store files that were not downloaded.
        # should have the three romanian data sets,
        # and the train, dev german data sets
        self.failed_loadings = []

        # store information about smallest data sets:
        # set_name : (lang, size)
        self.smallest_set = {
            "train" : (None, np.inf),
            "dev" : (None, np.inf),
            "test" : (None, np.inf)
        }

        # store file suffixes for train, dev, and test data sets
        data_sets = [
            "",
            "-train.iob2",
            "-dev.iob2",
            "-test.iob2"
        ]

        # initialize dict to store language Dataset objects
        self.lang2data = {}

        # iterate through languages
        for lang, url in lang2link.items():
            
            if verbose:
                print(f"\nLoading '{lang}' data ...")

            # iterate through data files (train, dev, test)
            lang_datasets = []
            for data_set in data_sets:

                # obtain complete file link and attempt download
                file_url = url + data_set
                sentences, labels, status_code = self._load_iob(file_url)

                # if download fails on any of train, dev, test sets, store the info
                if status_code != 200:
                    if data_set != "": 
                        self.failed_loadings.append((lang, data_set))
                
                # successful download
                else:
                    if verbose:
                        print(f"Successfully downloaded '{data_set}' data. ({len(sentences)} sentences)")
                    
                    # tokenize sentences, align labels, obtain and store Dataset
                    lang_data = self._create_Dataset(sentences, labels, self.model, verbose)
                    lang_datasets.append(lang_data)
            
            if verbose:
                print(f"Total data sets for '{lang}': {len(lang_datasets)}.")
            
            # update smallest data set data if needed
            for i, (set_name, (_, set_size)) in enumerate(self.smallest_set.items()):
                if (len(lang_datasets) >= i+1) and (set_size > lang_datasets[i].shape[0]):
                    self.smallest_set[set_name] = (lang, lang_datasets[i].shape[0])

                    if verbose:
                        print(f"New smallest {set_name} set: {lang_datasets[i].shape[0]} sentences ({lang}).")

            # link all downloaded datasets to proper language
            self.lang2data[lang] = lang_datasets

    def get_lang_data(self, lang: str) -> list[Dataset]:
        """
        Retrieve the list containing Dataset objects for the specified language code.
        Consult the supported language codes in the class documentation.

        Parameters
        ----------

        lang: str
            The language code corresponding to the needed Dataset objects.
        
        Returns
        -------

        lang_data: list[Dataset]
            The list containing the Dataset objects for the specified language.
        """

        # return list of Dataset objects for the given language
        return self.lang2data[lang]
    
    def get_smallest_set_size(self, set: Literal["train", "dev", "test", "all"] = 'all') -> tuple[str, Dataset] | dict[str, tuple[str, Dataset]]:
        """
        Retrieve the size of the smallest set of the specified type, together with its language code.
        It is possible to ask for the train, dev, or test sets, 
        as well as all of them, in which case a dictionary will be returned.

        Parameters
        ----------

        set: Literal["train", "dev", "test", "all"], optional (default= "all")
            The string associated with the specified data set type.
        
        Returns
        -------

        smallest_set_info : tuple[str, Dataset] | dict[str, tuple[str, Dataset]]
            The tuple (in case of a single set) containing (language, size),
            or the dictionary containing set_type - (language, size) pairs,
            with the needed data about the smallest recorded sets.
        """

        # return info about the smallest size for each set
        if set == 'all':
            return self.smallest_set
        
        # return info about the smallest specified set
        else:
            return self.smallest_set[set]
    
    def _load_iob(self, url: str) -> tuple[list[list[str]], list[list[str]]]:
        """
        Internal function: load the file and process it into two lists of lists.
        Both contain one list per sentence, with the first having words as individual entries,
        the other having the associated labels as individual entries.

        Returns sentence and labels lists, as well as the response code for the file request.
        """
        
        # obtain site response (data)
        response = requests.get(url)

        # initialize lists to store sentences and relative labels
        sentences = [[]]
        labels = [[]]

        # successful download
        if response.status_code == 200:

            # iterate through file lines
            for line in response.iter_lines():

                # end of sentence line: add new sentence list
                if str(line) == "b''" or str(line) == "\n":
                    sentences.append([])
                    labels.append([])
                
                # if non-empty line and not a comment
                elif (str(line)[0] != '#') and (str(line)[0:3] != "b'#") and (str(line)[0:3] != """b"#"""):

                    # split line: word_number, word, label_1, label_2, label_3
                    split_line = str(line.decode('utf-8')).split('\t')
                    word = split_line[1]
                    label = split_line[2]

                    # store word and label
                    sentences[-1].append(word)
                    labels[-1].append(label)
        
        # remove redundant new sentences (if file ends in multiple new lines)  
        while sentences[-1] == [] and len(sentences) > 1:
            sentences = sentences[:-1]
            labels = labels[:-1]
        
        return sentences, labels, response.status_code
    
    def _create_Dataset(self, sentences: list, labels: list, model: str, verbose: bool) -> Dataset:
        """
        Internal function: given the lists containing sentences and labels,
        tokenizes the sentences using the pre trained model's tokenizer,
        then aligns the labels to the new tokens.
        A Dataset object is constructed with tokens and aligned labels.
        """
        
        # obtain model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        # tokenize sentences
        tokenized = tokenizer(
            sentences,
            is_split_into_words=True,
            max_length=64,
            truncation="only_first", # Keep the beginning, slice the rest
            padding="max_length",    # Ensure all chunks are exactly 128
            return_overflowing_tokens=False,
            return_tensors="pt"      # Returns PyTorch tensors (or "tf", "np")
        )

        if verbose:
            # verify wether any sentence is truncated due to the set max_length
            for i, overflow in enumerate(tokenized.get("num_truncated_tokens", [])):
                if overflow > 0:
                    print(f"Sentence {i} lost {overflow} tokens due to truncation.")
        
        # align the labels to the tokenized sentences
        tokenized = self._align_labels(tokenized, labels)

        # create and return Dataset
        return Dataset.from_dict({k: v for k, v in tokenized.items()})
    
    
    def _align_labels(self, tokenized, labels):
        """
        Internal function: initialize a tensor with the shape of the token tensor,
        and populate it according to the type of tokens encountered.
        Only beginning-of-word tokens are linked to the relative label,
        while special tokens, padding, and subword tokens are going to be 'ignored'.
        """

        # create new list to store new labels
        all_labels = torch.empty_like(tokenized["input_ids"], dtype=torch.int8)

        # iterate through sentences
        for i in range(all_labels.shape[0]):
            word_ids = tokenized.word_ids(batch_index = i)

            # word_id tracker to properly mark subword tokens
            prev_word_id = None

            # iterate through words
            for j, word_id in enumerate(word_ids):

                # no word_id (special tokens, padding) -> ignore label
                if word_id is None:
                    all_labels[i, j] = -100
                
                # same word_id as previous token (subword token) -> ignore label
                elif word_id == prev_word_id:
                    all_labels[i, j] = -100
                
                # proper word token / first subword token -> obtain label id
                else:
                    all_labels[i, j] = self.tag2idx[labels[i][word_id]]

                # update last word id
                prev_word_id = word_id

        # add new labels to the tokenized sentences
        tokenized["labels"] = all_labels

        return tokenized  