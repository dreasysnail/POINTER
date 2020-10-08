import pke
import subprocess as sp
import os
import argparse
import glob
from tqdm import tqdm
from pytorch_transformers.tokenization_bert import BertTokenizer
from nltk.corpus import stopwords
import pdb
import re
import random
import yake


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

parser = argparse.ArgumentParser(description="""
Given `input_dir`, create eval directory under it.
for each *.predicted.txt under `input_dir`, evaluate it, push the evaluation result to files under eval directory.
The file name is *.eval.txt
""")
parser.add_argument("--file", type=str)
parser.add_argument("--n_keys", type=int, default=3)
parser.add_argument("--bert_model", type=str, default = "bert-base-uncased") 

args = parser.parse_args()

pos = {'NOUN', 'PROPN', 'ADJ'}

tokenizer = BertTokenizer.from_pretrained(args.bert_model) 

vocab_list = list(tokenizer.vocab.keys())

all_keys = []

with open(args.file, 'r+', encoding="utf-8") as f:
    with open(args.file[:-3] + 'key.txt', 'w', encoding="utf-8") as f_out:
        language = "en"
        max_ngram_size = 1
        deduplication_thresold = 0.9
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = args.n_keys + 2
        stoplist = set(stopwords.words('english')+ ['.', ',', '!', '(', ')', 'n\'t', '\'s', '-', '...', '``', '\'\'', '\'', ':', '?', '--', ';', '#', '/', '%', '[', ']', '{', '}'])
        for l in tqdm(f.read().splitlines()):
            l = l.lower()
            l = re.sub('\t',' ',l)
            custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
            keywords = custom_kw_extractor.extract_keywords(l)
            keyphrases = set([k[0] for k in keywords])
            keyphrases_ordered = []
            for k in l.split():
                if k in keyphrases:
                    keyphrases_ordered.append(k)
            save_list = [k for k in keyphrases_ordered if k in vocab_list and not is_number(k) and k not in stoplist]
            sampled_list = [save_list[i] for i in sorted(random.sample(range(len(save_list)), min(args.n_keys,len(save_list))))]
            f_out.write(" ".join(sampled_list) + '\n')
           
