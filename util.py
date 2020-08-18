import os
import logging
import torch
from collections import defaultdict
import re
from nltk.corpus import stopwords
import numpy as np


MAX_TURN = 6
PREVENT_FACTOR = 0.3
PROMOTE_FACTOR = 1.1
PREVENT_LIST = ['[UNK]', '"',"(",")","-","[","]","'","&"]  
STOP_LIST = set([s.lower for s in stopwords.words('english')]) | set(['[SEP]', '[PAD]', '[CLS]', 'the', 'of', 'and', 'in', 'a', 'to', 'was', 'is', '"', 'for', 'on', 'as', 'with', 'by', 'he', "'s", 'at', 'that', 'from', 'it', 'his', 'an', 'which', 's', '.', ',', '(', ')',"'", '%'])
REDUCE_LIST = set(["'",'s','.',","]) 

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'