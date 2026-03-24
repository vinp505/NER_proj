"""
Helper script to define classes related to data loading, processing and handling.
"""

# ------------------------------------------------------------

import requests
import torch
import numpy as np

from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets
from typing import Literal

# ------------------------------------------------------------
class LanguageData():
    """
    Class to load and store Dataset objects for each language.
    Given the language code, its associated entry in the lang2data dictionary is
    a list of Dataset objects.
    For standard language data, this list consists of, in order, a train, a dev and a test set.
    Romanian and German are exceptions, with only one dataset each.
    Informations about the smallest data sets among all languages, and among specified target languages, are also stored.

    The language codes are associated with the corresponding language in the following tuples: 
    ("slk", Slovakian), ("eng", English), ("swe", Swedish), ("nor", Norwegian), ("heb", Hebrew), ("rom", Romanian),
    ("por", Portuguese), ("ger", German), ("chi", Chinese), ("hrv", Croatian), ("srb", Serbian), ("dan", Danish).

    Parameters
    -----------

    model : str
        The string containing the base pretrained model name,
        needed to retrieve the tokenizer.
    
    target_langs : list, optional (default= ["eng", "slk", "dan", "rom", "chi", "heb"])
        List storing language codes for target languages of future fine-tuning.
        Information about the smallest target language data sets will be computed.

    verbose : bool, optional (default= False)
        Flag to signal wether or not the class should print
        information about the data loading process.
    
    Methods
    --------

    **get_lang_data(lang)**
        Retrieves the list containing Dataset objects for the specified language.
    
    **get_smallest_set_size(set)**
        Retrieves the size of the smallest set of the specified type across all languages, together with its language code.
        It is possible to ask for the train or test sets, 
        as well as both, in which case a dictionary will be returned.

    **get_smallest_target_set_size(set)**
        Retrieves the size of the smallest set of the specified type across target languages, together with its language code.
        It is possible to ask for the train or test sets, 
        as well as both, in which case a dictionary will be returned.
    """

    # model to obtain tokenizer
    def __init__(self, model: str, target_langs: list = ["eng", "slk", "dan", "rom", "chi", "heb"], verbose: bool = False):

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

        # store language codes and target languages
        self.lang_codes = ["slk", "eng", "swe", "nor", "heb", "rom", "por", "ger", "chi", "hrv", "srb", "dan"]
        self.target_langs = target_langs

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
            "test" : (None, np.inf)
        }

        # store information about smallest data sets among target languages:
        # set_name : (lang, size)
        self.smallest_set_tl = {
            "train" : (None, np.inf),
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

            # if only one data set was loaded, split into train and test
            if len(lang_datasets) == 1:

                if verbose:
                   print(f"\nOnly one file successfully loaded for '{lang}'. It will be split into train (80%) and test (20%) sets")

                # 80 - 20 split
                d = lang_datasets[0]
                train_set_size = int(0.8*d.shape[0])
                train_set = d.select(range(train_set_size))
                test_set = d.select(range(train_set_size, d.shape[0]))

                # update language datasets
                lang_datasets = [train_set, test_set]

                if verbose:
                    print(f"Successfully created train ({train_set.shape[0]} sentences) and test ({test_set.shape[0]} sentences) sets for '{lang}'.")
            
            # if all three data sets were loaded, merge train and dev sets
            if len(lang_datasets) == 3:

                if verbose:
                    print(f"\nAll three files successfully loaded for '{lang}'. Train and dev sets will be merged.")
                
                # concatenate train and dev sets, and update language datasets
                train_dev_set = concatenate_datasets([lang_datasets[0], lang_datasets[1]], axis= 0)
                lang_datasets = [train_dev_set, lang_datasets[2]]

                if verbose:
                    print(f"Successfully merged data sets. New training set contains {train_dev_set.shape[0]} sentences.")

            # update smallest data set data (target and overall) if needed
            if lang in self.target_langs:
                for i in range(2):
                    set_name, (_, set_size) = list(self.smallest_set.items())[i]
                    _, (_, set_size_target) = list(self.smallest_set_tl.items())[i]

                    if (len(lang_datasets) >= i+1) and (set_size > lang_datasets[i].shape[0]):
                        self.smallest_set[set_name] = (lang, lang_datasets[i].shape[0])

                        if verbose:
                            print(f"New smallest {set_name} set: {lang_datasets[i].shape[0]} sentences ({lang}).")
                    
                    if (len(lang_datasets) >= i+1) and (set_size_target > lang_datasets[i].shape[0]):
                        self.smallest_set_tl[set_name] = (lang, lang_datasets[i].shape[0])

                        if verbose:
                            print(f"New smallest target language {set_name} set: {lang_datasets[i].shape[0]} sentences ({lang}).")
            
            # update overall smallest data set data if needed
            else:
                for i, (set_name, (_, set_size)) in enumerate(self.smallest_set.items()):
                        if (len(lang_datasets) >= i+1) and (set_size > lang_datasets[i].shape[0]):
                            self.smallest_set[set_name] = (lang, lang_datasets[i].shape[0])

                            if verbose:
                                print(f"New smallest {set_name} set: {lang_datasets[i].shape[0]} sentences ({lang}).")

            # link all downloaded datasets to proper language
            self.lang2data[lang] = lang_datasets

    def get_lang_data(self, lang: str) -> list[Dataset]:
        """
        Retrieves the list containing Dataset objects for the specified language code.
        Consult the supported language codes in the class documentation.

        Parameters
        ----------

        lang : str
            The language code corresponding to the needed Dataset objects.
        
        Returns
        -------

        lang_data : list[Dataset]
            The list containing the Dataset objects for the specified language.
        """

        # return list of Dataset objects for the given language
        return self.lang2data[lang]
    
    def get_smallest_set_size(self, set: Literal["train", "test", "all"] = 'all') -> tuple[str, Dataset] | dict[str, tuple[str, Dataset]]:
        """
        Retrieves the size of the smallest set of the specified type across all languages, together with its language code.
        It is possible to ask for the train or test sets, 
        as well as both, in which case a dictionary will be returned.

        Parameters
        ----------

        set : Literal["train", "test", "all"], optional (default= "all")
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
        
    def get_smallest_target_set_size(self, set: Literal["train", "test", "all"] = 'all') -> tuple[str, Dataset] | dict[str, tuple[str, Dataset]]:
        """
        Retrieves the size of the smallest set of the specified type across target languages, together with its language code.
        It is possible to ask for the train or test sets, 
        as well as both, in which case a dictionary will be returned.

        Parameters
        ----------

        set : Literal["train", "test", "all"], optional (default= "all")
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
            return self.smallest_set_tl
        
        # return info about the smallest specified set
        else:
            return self.smallest_set_tl[set]
    
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

# ------------------------------------------------------------

class DataSplit():
    """
    Class to load and store the unique train set, and one test set per language.
    If no specific target language is provided, the train set will contain an equal amount of sequences
    for each language.
    If, instead, a target language is specified, the train set will contain k sequences for other languages,
    and an amount of sequences for the target language equal to the smallest recorded train set.

    Parameters
    -----------

    langData : LanguageData
        The object containing data sets for each language.
    
    target_lang : str, optional (default= "all")
        Language code of the target language.
        By default, it is set to "all", resulting in an equally split train set.

    k : int, optional (default= 10)
        Amount of sequences to select from the train set of non-target languages.
        Must be less than or equal to the size of the smallest train set recorded.

    random_state : int | None, optional (default= None)
        Random state seed to ensure deterministic train and test sets.
    
    Methods
    --------

    **get_train_set()**
        Retrieves the train set.
    
    **get_test_set(lang)**
        Retrieves the dictionary containing lang - test_set pairs, or,
        if a specific language is given, the relative test set Dataset object.
    """
        
    def __init__(self, langData: LanguageData, target_lang: str = "all", k: int = 10, random_state: int | None = None):
        
        # ensure the target language is among the ones considered target language when loading the data
        assert (target_lang in langData.target_langs) or target_lang == 'all', f"The given target language {target_lang} is not present in the accepted target language list of the LanguageData object {langData.target_langs}"
        
        # ensure the amount of sentences for non-target languages isn't greater than the smallest test set
        assert k <= langData.get_smallest_set_size("test")[1], f"The given k is greater than the size of the smallest test set ({k} > {langData.get_smallest_set_size("test")[1]})"
        
        # store information
        self.langData = langData
        self.target_lang = target_lang
        self.k = k
        self.random_state = random_state

        # initialize train set, and test set dictionary
        self.train_set = []
        self.test_sets = {}

        # iterate through the languages
        for lang in self.langData.lang_codes:
            
            # obtain train and test sets
            train_set_lang = self._obtain_train_set(lang)
            test_set = self._obtain_test_set(lang)

            # store data
            self.train_set.append(train_set_lang)
            self.test_sets[lang] = test_set
        
        # only one train set is needed, concatenate individual sets into one
        self.train_set = concatenate_datasets(self.train_set, axis= 0)


    def get_train_set(self) -> Dataset:
        """
        Retrieves the train set.

        Returns
        -------

        train_set : Dataset
            The train set.
        """
        return self.train_set
    
    def get_test_set(self, lang: str = 'all') -> dict[str, Dataset] | Dataset:
        """
        Retrieves the dictionary containing lang - test_set pairs, or,
        if a specific language is given, the relative test set Dataset object.

        Parameters
        ----------

        lang : str, optional (default= 'all')
            The language whose test set is to be retrieved.
            By default, the dictionary containing lang - test_set pairs
            for each language will be returned instead.
        
        Returns
        -------

        test_set : dict[str, Dataset] | Dataset
            If no language is specified, the dictionary containing lang - test_set pairs
            for each language.
            Else, the language specific test set Dataset object.
        """

        # return entire dictionary
        if lang == 'all':
            return self.test_sets
        
        # return language specific test set
        return self.test_sets[lang]


    def _obtain_train_set(self, lang) -> Dataset:
        """
        Internal function: retrieve language train set,
        shrink it according to language.
        For non-target languages, only retain k sequences.
        """

        # retrieve language train set
        lang_data = self.langData.get_lang_data(lang)
        train_set = lang_data[0]

        # no target language -> all train sets of equal proportions
        if self.target_lang == "all":

            # redute train set to size of the smallest train set
            smallest_train_set = self.langData.get_smallest_set_size("train")[1]
            sampled_train_set = train_set.shuffle(seed= self.random_state).select(range(smallest_train_set))
        
        # target language
        elif lang == self.target_lang:

            # reduce train set to the size of the smallest target language train set
            smallest_target_train_set = self.langData.get_smallest_target_set_size("train")[1]
            sampled_train_set = train_set.shuffle(seed= self.random_state).select(range(smallest_target_train_set))
        
        # not a target language
        else:

            # reduce train set to contain k samples
            sampled_train_set = train_set.shuffle(seed= self.random_state).select(range(self.k))
        
        return sampled_train_set


    def _obtain_test_set(self, lang) -> Dataset:
        """
        Internal function: retrieve language test set,
        and resize it to have the same amount of sequences
        as the smallest test set.
        """
        
        # retrieve language test set
        lang_data = self.langData.get_lang_data(lang)
        test_set = lang_data[1]

        # reduce test set to size of the smallest train set
        smallest_test_set = self.langData.get_smallest_set_size("test")[1]
        sampled_test_set = test_set.shuffle(seed= self.random_state).select(range(smallest_test_set))
        
        return sampled_test_set


