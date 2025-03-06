
from OpenAttack.text_process.constituency_parser.base import ConstituencyParser
from OpenAttack.tags import *
from OpenAttack.data_manager import DataManager
from pathlib import Path
import os
import pickle

# CACHE_FILE = str('cache.pkl')

# def load_parser():
#     if os.path.exists(CACHE_FILE):
#         with open(CACHE_FILE, 'rb') as f:
#             parser = pickle.load(f)
#     else:
#         parser = DataManager.load("TProcess.StanfordParser")
#         with open(CACHE_FILE, 'wb') as f:
#             pickle.dump(parser, f)
#     return parser

class StanfordParser(ConstituencyParser):
    """
    Constituency parser based on stanford parser.

    :Requirements:
        * java

    """

    TAGS = {TAG_English}

    def __init__(self):
        self.__parser = DataManager.load("TProcess.StanfordParser")

    def parse(self, sentence: str) -> str:
        return str(list(self.__parser(sentence))[0])