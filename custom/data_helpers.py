import requests

class LanguageData():

    def __init__(self):
        self.lang2link = {
            "slk" : "https://github.com/UniversalNER/UNER_Slovak-SNK",
            "eng" : "https://github.com/UniversalNER/UNER_English-EWT",
            "swe" : "https://github.com/UniversalNER/UNER_Swedish-Lines",
            "nor" : "https://github.com/UniversalNER/UNER_Norwegian-NDT",
            "heb" : "https://github.com/UniversalNER/UNER_Hebrew-HTB",
            "rom" : "https://github.com/UniversalNER/UNER_Romanian-LegalNERo",
            "por" : "https://github.com/UniversalNER/UNER_Portuguese-Bosque",
            "ger" : "https://github.com/UniversalNER/UNER_German-PUD",
            "chi" : "https://github.com/UniversalNER/UNER_Chinese-GSD",
            "hrv" : "https://github.com/UniversalNER/UNER_Croatian-SET",
            "srb" : "https://github.com/UniversalNER/UNER_Serbian-SET",
            "dan" : "https://github.com/UniversalNER/UNER_Danish-DDT"
        }
    
    def retrieve_language_data(self, lang: str):
        ...


    