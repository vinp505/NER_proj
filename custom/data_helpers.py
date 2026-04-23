"""
Helper script to define classes related to data loading, processing and handling.
"""

# ------------------------------------------------------------

import requests
import random
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

    The original dataset used is UniversalNER https://github.com/UniversalNER.

    Parameters
    -----------

    model : str
        The string containing the base pretrained model name,
        needed to retrieve the tokenizer.
    
    target_langs : list | str, optional (default= ["eng", "slk", "dan", "rom", "chi"])
        List storing language codes for target languages of future fine-tuning.
        Information about the smallest target language data sets will be computed.
        It is possible to specify 'all' as a value, as a shorthand for adding all language codes in the list.

    verbose : bool, optional (default= False)
        Flag to signal wether or not the class should print
        information about the data loading process.
    
    Methods
    --------

    **get_lang_data(lang)**
        Retrieves the list containing Dataset objects for the specified language.

    **get_lang_sentences(lang)**
        Retrieves the list containing sentences (split into words) for the specified language.
    
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
    def __init__(self, model: str, target_langs: list | str = ["eng", "slk", "dan", "rom", "chi"], verbose: bool = False):

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
        
        if isinstance(target_langs, str):
            target_langs = [target_langs.lower()]

        if target_langs == ['all']:
            target_langs = self.lang_codes

        self.target_langs = target_langs

        # store model and entity tags conversion dictionaries
        self.model = model
        self.tag2idx = {
            "O": 0, 
            'B-ORG': 1, 'I-ORG': 2,  
            'B-OTH': 3, 'I-OTH': 4, 'I-PER': 3, 'B-PER': 4,  # for an out-of-insturctions tag 'OTHER', convert to PERSON 
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
        self.smallest_target_set = {
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

        # initialize dict to store language sentences
        self.lang2sent = {}

        # iterate through languages
        for lang, url in lang2link.items():
            
            if verbose:
                print(f"\nLoading '{lang}' data ...")


            # set up collections for data

            lang_datasets = {
                "train" : None,
                "test" : None
            }

            lang_sentences = {
                "train" : None,
                "test" : None
            }

            tmp_datasets = []
            tmp_sentences = []

            # iterate through data files (train, dev, test)
            for data_set in data_sets:

                # obtain complete file link and attempt download
                file_url = url + data_set
                raw_sentences, sentences, labels, status_code = self._load_iob(file_url)

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
                    tmp_datasets.append(lang_data)
                    tmp_sentences.append(raw_sentences)
            
            if verbose:
                print(f"Total data sets for '{lang}': {len(tmp_datasets)}.")

            # if only one data set was loaded, split into train and test
            if len(tmp_datasets) == 1:
                
                # obtain proper train set fraction based on the language
                tr = 0.8 if lang in self.target_langs else 0.5
                
                if verbose:
                   print(f"\nOnly one file successfully loaded for '{lang}'. It will be split into train ({tr*100}%) and test ({(1-tr)*100}%) sets.")

                
                # split
                
                d = tmp_datasets[0]
                train_set_size = int(tr*d.shape[0])
                train_set = d.select(range(train_set_size))
                test_set = d.select(range(train_set_size, d.shape[0]))

                s = tmp_sentences[0]
                train_sentences = s[:train_set_size]
                test_sentences = s[train_set_size:]


                # update language datasets
                lang_datasets["train"] = train_set
                lang_datasets["test"] = test_set

                lang_sentences["train"] = train_sentences
                lang_sentences["test"] = test_sentences

                if verbose:
                    print(f"Successfully created train ({train_set.shape[0]} sentences) and test ({test_set.shape[0]} sentences) sets for '{lang}'.")

            # if only two files were loaded, use for train, and test
            if len(tmp_datasets) == 2:

                lang_datasets["train"] = tmp_datasets[0]
                lang_datasets["test"] = tmp_datasets[1]

                lang_sentences["train"] = tmp_sentences[0]
                lang_sentences["test"] = tmp_sentences[1]

            # if all three data sets were loaded, merge appropriate data sets
            if len(tmp_datasets) == 3:

                if verbose:
                    print(f"\nAll three files successfully loaded for '{lang}'. Data will be merged.")
                
                # train and test sets are needed
                train_dev_set = concatenate_datasets([tmp_datasets[0], tmp_datasets[1]], axis= 0)
                train_dev_sentences = tmp_sentences[0] + tmp_sentences[1]

                lang_datasets["train"] = train_dev_set
                lang_datasets["test"] = tmp_datasets[2]
  
                lang_sentences["train"] = train_dev_sentences
                lang_sentences["test"] = tmp_sentences[2]

                if verbose:
                    print(f"Successfully merged train and dev sets. New training set contains {lang_datasets["train"].shape[0]} sentences.")

            # update smallest data set data (target and overall) if needed
            # check train and test set dimensions
            for set_name in ["train", "test"]:

                # obtain current smallest set sizes
                set_size = self.smallest_set[set_name][1]
                set_size_target = self.smallest_target_set[set_name][1]


                # update if needed

                if set_size > lang_datasets[set_name].shape[0]:
                    self.smallest_set[set_name] = (lang, lang_datasets[set_name].shape[0])

                    if verbose:
                        print(f"New smallest {set_name} set: {lang_datasets[set_name].shape[0]} sentences ({lang}).")
                
                if set_size_target > lang_datasets[set_name].shape[0] and lang in self.target_langs:
                    self.smallest_target_set[set_name] = (lang, lang_datasets[set_name].shape[0])

                    if verbose:
                        print(f"New smallest target language {set_name} set: {lang_datasets[set_name].shape[0]} sentences ({lang}).")

            # link all downloaded datasets to proper language
            self.lang2data[lang] = lang_datasets
            self.lang2sent[lang] = lang_sentences

    def get_lang_data(self, lang: str) -> dict[str, Dataset]:
        """
        Retrieves the dictionary containing Dataset objects for the specified language code.
        Consult the supported language codes in the class documentation.

        Parameters
        ----------

        lang : str
            The language code corresponding to the needed Dataset objects.
        
        Returns
        -------

        lang_data : dict[Dataset]
            The dictionary containing the Dataset objects for the specified language.
        """

        # return list of Dataset objects for the given language
        return self.lang2data[lang]

    def get_lang_sentences(self, lang: str) -> dict[str, list[list[str]]]:
        """
        Retrieves the dictionary containing metadata and individual words for each sentence for the specified language code.
        Consult the supported language codes in the class documentation.

        Parameters
        ----------

        lang : str
            The language code corresponding to the needed sentence data.
        
        Returns
        -------

        lang_sentences : dict[list[list[str]]]
            The dictionary containing the Dataset objects for the specified language.
        """

        # return list of Dataset objects for the given language
        return self.lang2sent[lang]
    
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
            return self.smallest_target_set
        
        # return info about the smallest specified set
        else:
            return self.smallest_target_set[set]
    
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
        raw_sentences = [[]]  # stores metadata about the sentence (ID, text)
        sentences = [[]]
        labels = [[]]

        # successful download
        if response.status_code == 200:

            # iterate through file lines
            for line in response.iter_lines():

                # end of sentence line: add new sentence list
                if str(line) == "b''" or str(line) == "\n":
                    raw_sentences.append([])
                    sentences.append([])
                    labels.append([])

                # if not empty line but a comment
                elif (str(line)[0] == '#') or (str(line)[0:3] == "b'#") or (str(line)[0:3] == """b"#"""):
                    raw_sentences[-1].append(str(line.decode('utf-8')))

                # annotated word
                else:
                    # split line: word_number, word, label_1, label_2, label_3
                    split_line = str(line.decode('utf-8')).split('\t')
                    word = split_line[1]
                    label = split_line[2]

                    # store word and label
                    raw_sentences[-1].append(word)
                    sentences[-1].append(word)
                    labels[-1].append(label)
        
        # remove redundant new sentences (if file ends in multiple new lines)  
        while sentences[-1] == [] and len(sentences) > 1:
            raw_sentences = raw_sentences[:-1]
            sentences = sentences[:-1]
            labels = labels[:-1]
        
        return raw_sentences, sentences, labels, response.status_code
    
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
    
    Methods
    --------

    **get_train_set()**
        Retrieves the train set.
    
    **get_train_sentences()**
        Retrieves the train sentences list.
    
    **get_test_set(lang)**
        Retrieves the dictionary containing lang - test_set pairs, or,
        if a specific language is given, the relative test set Dataset object.

    **get_test_sentences(lang)**
        Retrieves the dictionary containing lang - test_sentences pairs, or,
        if a specific language is given, the relative list of test senteces.
    """
        
    def __init__(self, langData: LanguageData, target_lang: str = "all", k: int = 10):
        
        # ensure the target language is among the ones considered target language when loading the data
        assert (target_lang in langData.target_langs) or target_lang == 'all', f"The given target language {target_lang} is not present in the accepted target language list of the LanguageData object {langData.target_langs}"
        
        # ensure the amount of sentences for non-target languages isn't greater than the smallest test set
        assert k <= langData.get_smallest_set_size("test")[1], f"The given k is greater than the size of the smallest test set ({k} > {langData.get_smallest_set_size("test")[1]})"
        
        # store information
        self.langData = langData
        self.target_lang = target_lang
        self.k = k

        # initialize train set, and test set dictionary
        self.train_set = []
        self.test_sets = {}

        self.train_sent = []
        self.test_sent = {}

        # iterate through the languages
        for lang in self.langData.lang_codes:
            
            # obtain train and test sets
            train_set_lang, train_sent_lang = self._obtain_train_set(lang)
            test_set, test_sent = self._obtain_test_set(lang)

            # store data (if any was needed)
            if train_sent_lang:
                self.train_set.append(train_set_lang)
                self.test_sets[lang] = test_set

            self.train_sent.extend(train_sent_lang)
            self.test_sent[lang] = test_sent
        
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
    
    def get_train_sentences(self) -> list[list[str]]:
        """
        Retrieves the train sentences.

        Returns
        -------

        train_sent : list[list[str]]
            The list of train sentences split into lists of words.
        """
        return self.train_sent
    
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
    
    def get_test_sentences(self, lang: str = 'all') -> dict[str, list[list[str]]] | list[list[str]]:
        """
        Retrieves the dictionary containing lang - test_sentences pairs, or,
        if a specific language is given, the relative list of test senteces.

        Parameters
        ----------

        lang : str, optional (default= 'all')
            The language whose test set is to be retrieved.
            By default, the dictionary containing lang - test_set pairs
            for each language will be returned instead.
        
        Returns
        -------

        test_sent : dict[str, list[list[str]]] | list[list[str]]
            If no language is specified, the dictionary containing 
            lang - test_sentences pairs for each language.
            Else, the language specific list of test senteces.
        """

        # return entire dictionary
        if lang == 'all':
            return self.test_sent
        
        # return language specific test set
        return self.test_sent[lang]


    def _obtain_train_set(self, lang) -> Dataset:
        """
        Internal function: retrieve language train set,
        shrink it according to language.
        For non-target languages, only retain k sequences.
        """

        # retrieve language train set
        lang_data = self.langData.get_lang_data(lang)
        train_set = lang_data["train"]

        lang_sent = self.langData.get_lang_sentences(lang)
        train_sent = lang_sent["train"]

        # no target language -> all train sets of equal proportions
        # redute train set to size of the smallest train set
        if self.target_lang == "all":
            size = self.langData.get_smallest_set_size("train")[1]
        
        # target language
        # reduce train set to the size of the smallest target language train set
        elif lang == self.target_lang:
            size = self.langData.get_smallest_target_set_size("train")[1]
        
        # not a target language
        # reduce train set to contain k samples
        else:
            size = self.k
        
        # do not load anything if not required
        if size == 0:
            return False, False

        # obtain data sample
        sampled_train_set, sampled_train_sent = self._sample_data(train_set, train_sent, size)
        return sampled_train_set, sampled_train_sent


    def _obtain_test_set(self, lang: str) -> Dataset:
        """
        Internal function: retrieve language test set,
        and resize it to have the same amount of sequences
        as the smallest test set.
        """
        
        # retrieve language test set
        lang_data = self.langData.get_lang_data(lang)
        test_set = lang_data["test"]

        lang_sent = self.langData.get_lang_sentences(lang)
        test_sent = lang_sent["test"]

        # reduce test set to size of the smallest train set
        size = self.langData.get_smallest_set_size("test")[1]
        sampled_test_set, sampled_test_sent = self._sample_data(test_set, test_sent, size)
        
        return sampled_test_set, sampled_test_sent
    
    def _sample_data(self, data: Dataset, sentences: list[list[str]], size: int) -> Dataset:
        """
        Internal function: obtain a sample of the given data of the given size,
        ensuring entity density is preserved.
        """

        # if size is equal to amt of data, just return the data
        if size == data.shape[0]:
            return data, sentences
        
        # compute entity density by counting total amt of tokens and of entities
        tot_tokens = sum([len(sample["labels"]) for sample in data])  # !!! THIS COUNTS END OF SENTENCE PADDING TOKENS BUT IT IS FINE
        tot_entities = sum([sum([1 for x in sample["labels"] if (x != -100) and (x != 0)]) for sample in data])#sum up all the entities
        density = tot_entities / tot_tokens

        # obtain amount of required densities to preserve density
        req_entities = int(size * 64 * density)

        print(f"density: {density}")
        print(f"req_entities: {req_entities}")

        # split data into sentences containing entities and sentences not containing any
        dataset_ent = data.filter(lambda row: any((tag != -100) and (tag != 0) for tag in row["labels"]))
        dataset_no_ent = data.filter(lambda row: all((tag == -100) or (tag == 0) for tag in row["labels"]))

        # initialize list to split data
        sent_ent = []
        sent_no_ent = []

        # keep track of total entities encountered and amount of sentences needed for it
        tot_entities = 0
        amt_needed_sentences = 0

        # iterate through data
        for i, sample in enumerate(data):
            
            # compute amount of entities in current sentence
            amt_entities = sum((tag != -100) and (tag != 0) for tag in sample["labels"])

            # if any entities are present
            if amt_entities:

                # store in appropriate list
                sent_ent.append(sentences[i])

                # if more entities are still needed update counters
                if tot_entities < req_entities:
                    amt_needed_sentences += 1
                    tot_entities += amt_entities
            
            # if no entities are present
            else:
                sent_no_ent.append(sentences[i])

        print(f"Sentences with entities: {len(sent_ent)}")
        print(f"Sentences with no entities: {len(sent_no_ent)}")
        print(f"Amt. needed sentences with entities: {amt_needed_sentences}")

        # only retain the amount of sentences needed to preserve the entity density
        sampled_data_ent = dataset_ent.select(range(amt_needed_sentences))
        sampled_sent_ent = sent_ent[:amt_needed_sentences]

        # the remaining available "slots" go to sentences with no labels
        sampled_data_no_ent = dataset_no_ent.select(range(size-amt_needed_sentences))
        sampled_sent_no_ent = sent_no_ent[:(size-amt_needed_sentences)]

        # merge back the two sampled data sets
        sampled_data = concatenate_datasets([sampled_data_ent, sampled_data_no_ent], axis= 0)
        sampled_sent = sampled_sent_ent + sampled_sent_no_ent
        

        # shuffle
        
        new_idx = [i for i in range(size)]
        random.shuffle(new_idx)

        sampled_data = sampled_data.select(new_idx)
        sampled_sent = [sampled_sent[i] for i in new_idx]

        return sampled_data, sampled_sent


