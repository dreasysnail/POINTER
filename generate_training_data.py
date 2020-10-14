from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool

from random import random, randrange, randint, shuffle, choice
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import json
import collections
from utils_ig import house_robber
from multiprocessing import Process, Value, Lock, Manager
import multiprocessing
from joblib import Parallel, delayed

import logging
import pdb
import os
from collections import Counter
from nltk.corpus import stopwords
import spacy
import yake
import re
from nltk.tokenize import TweetTokenizer
logger = logging.getLogger(__name__)

CUTOFF = 5

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    "Remove random truncate to remove last sentence"
    if len(tokens_a) <= max_num_tokens:
        return

    indices = [i for i, x in enumerate(tokens_a) if x == "."]
    
    if len(indices) > 0 and indices[-1] == len(tokens_a) - 1: #end with .
        del indices[-1]

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        if len(indices) == 0:
            return None
        else:
            del trunc_tokens[indices[-1]+1:]
            del indices[-1]
            

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def add_doc_parallel(line, docs, tokenizer):
    docs.add_document(tokenizer.tokenize(line))


def create_masked_lm_predictions(tokens, whole_word_mask, vocab_list, idf_dict, cls_token_at_end=False, pad_on_left=False,
                        cls_token='[CLS]', sep_token='[SEP]', noi_token='[NOI]', pad_token=0,
                        sequence_a_segment_id=0,
                        cls_token_segment_id=1, pad_token_segment_id=0,
                        mask_padding_with_zero=True,
                        token_value='idf', args=None):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    if args.wp:
        if not cls_token in tokens:
            return
        cls_pos = tokens.index(cls_token)
        
        tokens_src = tokens[:cls_pos]
        tokens = tokens[cls_pos:] 
        lm_label_tokens_src = [noi_token] * len(tokens_src)


    if token_value == 'idf':
        prob_list =  np.array([idf_dict[t] for t in tokens])
    else: # token_value == 'tf-idf' or token_value == 'tf-idf-stop':
        tf = Counter(tokens)
        tokens_len = float(len(tokens))
        # score: higher will be more likely to be keeped
        prob_list =  np.array([idf_dict[t] * tf[t] / tokens_len  for t in tokens]) 

    
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(" ".join(tokens))
    key_word_len = 2 #/len(keywords)
    for i, t in enumerate(tokens):
        if  t in keywords:
            prob_list[i] *= 2  
    # Repeat words
    for i, t in enumerate(tokens):
        if t in tokens[:i]:
            prob_list[i] /= 10


         
    # prob_list: now, lower will be more likely to be keeped
    prob_list = max(prob_list) - prob_list


    lm_label_tokens =  [noi_token] * len(tokens)

    # comment out for not generating all noi sequence
    if args.wp:
        yield tokens_src+tokens, lm_label_tokens_src + lm_label_tokens
    else:
        yield tokens, lm_label_tokens

    origin_tokens = tokens.copy()
    origin_prob_list = prob_list.copy()
    for skip in range(2,args.skip+1):
        N = len(origin_tokens)
        tokens = origin_tokens.copy()
        prob_list = origin_prob_list.copy()
        while N > key_word_len + skip:
            mask_pos = np.array(house_robber(prob_list,skip = skip))
            unmask_pos = np.setdiff1d(np.arange(N), mask_pos)

            lm_label_tokens = ['[PAD]'] * len(unmask_pos)
            j=0
            i = 1
            while i < len(prob_list):
                if i in mask_pos:
                    lm_label_tokens[j] = tokens[i]  
                    i += 2
                else:
                    lm_label_tokens[j] = noi_token
                    i += 1
                j += 1
            while j < len(unmask_pos):
                lm_label_tokens[j] = noi_token # no input for last token of new sequence
                j+= 1

            tokens = [t  for i,t in enumerate(tokens) if i in unmask_pos]
            N = len(tokens)

            prob_list = prob_list[unmask_pos]

            if args.wp:
                yield tokens_src + tokens, lm_label_tokens_src + lm_label_tokens
            else:
                yield tokens, lm_label_tokens

    
def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob, whole_word_mask, vocab_list, idf_dict,token_value=None, args=None):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    instances = []
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    
    tokens_a = document
    truncate_seq_pair(tokens_a, [], max_num_tokens)

    assert len(tokens_a) >= 1

    if args.wp:
        tokens =  tokens_a + ["[PAD]"]
    else:
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + ["[PAD]"]

    for tokens, lm_label_tokens in create_masked_lm_predictions(
                    tokens, whole_word_mask, vocab_list, idf_dict, token_value=token_value, args=args):

    
        instance = {
            "tokens": tokens,
            "lm_label_tokens": lm_label_tokens,
            }
        instances.append(instance)
    return instances


def create_training_file(docs, vocab_list, args, epoch_num, index_s, index_e, idf_dict, token_value):
    epoch_filename = args.output_dir / "epoch_{}.json".format(epoch_num)
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(index_s, index_e, desc="Document"):
            if len(docs[doc_idx]) <= CUTOFF: continue
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list, idf_dict=idf_dict, token_value=token_value, args=args)
            doc_instances = [json.dumps(instance) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_file = args.output_dir / "epoch_{}_metrics.json".format(epoch_num)
    with metrics_file.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))
    

def cal_idf(idf_dict, docs, index_s, index_e, lock=None, args=None):
    local_dict = {}
    logger.info("start partition at {}".format(index_s))
    for i in trange(index_s, index_e, ):
        if args.wp:
            if not "[CLS]" in docs[i]:
                continue
            cls_pos = docs[i].index("[CLS]")
            

            tokens=  docs[i][cls_pos:]  + ["[PAD]"]
        else:
            tokens= ["[CLS]"] + docs[i] + ["[SEP]"] + ["[PAD]"]
        for t in set(tokens):
            local_dict[t] = local_dict.get(t, 0) + 1
    if lock:
        with lock:
            for k, v in local_dict.items():
                idf_dict[k] = idf_dict.get(k,0) + v
    else:
        for k, v in local_dict.items():
            idf_dict[k] = idf_dict.get(k,0) + v

def clean_str(txt):
	#print("in=[%s]" % txt)
	txt = txt.lower()
	txt = re.sub('^',' ', txt)
	txt = re.sub('$',' ', txt)

	# url and tag
	words = []
	for word in txt.split():
		i = word.find('http') 
		if i >= 0:
			word = word[:i] + ' ' + '__url__'
		words.append(word.strip())
	txt = ' '.join(words)

	# remove markdown URL
	txt = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', txt)

	# remove illegal char
	txt = re.sub('__url__','URL',txt)
	txt = re.sub(r"[^A-Za-z0-9():,.!?\"\']", " ", txt)
	txt = re.sub('URL','__url__',txt)	

	# contraction
	add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
	tokenizer = TweetTokenizer(preserve_case=False)
	txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
	txt = txt.replace(" won't ", " will n't ")
	txt = txt.replace(" can't ", " can n't ")
	for a in add_space:
		txt = txt.replace(a+' ', ' '+a+' ')

	txt = re.sub(r'^\s+', '', txt)
	txt = re.sub(r'\s+$', '', txt)
	txt = re.sub(r'\s+', ' ', txt) # remove extra spaces
	
	return txt

def partitionIndexes(totalsize, numberofpartitions):
    # Compute the chunk size (integer division; i.e. assuming Python 2.7)
    chunksize = int(totalsize / numberofpartitions)
    # How many chunks need an extra 1 added to the size?
    remainder = totalsize - chunksize * numberofpartitions
    a = 0
    for i in range(numberofpartitions):
        b = a + chunksize + (i < remainder)
        # Yield the inclusive-inclusive range
        yield (a, b )
        a = b

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=1,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--duplicate_epochs", type=int, default=1,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--task_name", type=str, default='',
                        help="Dataset name")
    parser.add_argument("--token_value", type=str, default='df-stop',
                        help="Dataset name")
    parser.add_argument("--wp", type=bool, default=False, help="if train on wp")
    parser.add_argument("--max_line", type=int, default=None, help="maximum lines to process")
    parser.add_argument("--skip", type=int, default=2, help="minimal gap ranges")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument('--file_id_option', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO )
    
    if args.file_id_option:
        new_suffix = "." + ("%03d" %args.file_id_option)
        args.train_corpus = (args.train_corpus.parent / args.train_corpus.name).with_suffix(args.train_corpus.suffix + new_suffix)
        args.output_dir  = (args.output_dir.parent / args.output_dir.name).with_suffix(args.output_dir.suffix + new_suffix)
    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    if tokenizer._noi_token is None:
        tokenizer._noi_token = '[NOI]'
        if args.bert_model == 'bert-base-uncased':
            tokenizer.vocab['[NOI]'] = tokenizer.vocab.pop('[unused0]')
        elif args.bert_model == 'bert-base-cased':
            tokenizer.vocab['[NOI]'] = tokenizer.vocab.pop('[unused1]')
        else:
            raise ValueError("No clear choice for insert NOI for tokenizer type {}".format(args.model_name_or_path))
        tokenizer.ids_to_tokens[1] = '[NOI]'
        logger.info("Adding [NOI] to the vocabulary 1")

    vocab_list = list(tokenizer.vocab.keys())
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            lines_to_add = []
            iterations = 0
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                iterations += 1
                if args.max_line and iterations >= args.max_line:
                    break
                line = line.strip()
                if line == "" or len(line.split()) <=1:
                    continue
                else:
                    if args.task_name == 'yelp' or args.task_name == 'trip':
                        line = " ".join(line.split("\t"))
                    if args.clean:
                        line = clean_str(line)
                    lines_to_add.append(line)

            for line in tqdm(lines_to_add):

                tokens = line.split()
                docs.add_document(tokens)
                
                        
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        args.output_dir.mkdir(exist_ok=True)

        # count idf score 
        logger.info("Calculate idf score begin")
        if args.num_workers > 1:
            manager = Manager()
            idf_dict = manager.dict()
            lock = Lock()
            logger.info("Number of CPUs {}".format(multiprocessing.cpu_count()))
            idx_list = [i for i in partitionIndexes(len(docs), multiprocessing.cpu_count())]
            procs = [Process(target=cal_idf, args=(idf_dict, docs, idx_list[i][0], idx_list[i][1], lock)) for i in range(multiprocessing.cpu_count())]
            
            for p in procs: p.start()
            logger.info("processing finished")
            for p in procs: p.join()
     
        else:
            idf_dict = {}
            cal_idf(idf_dict, docs, 0, len(docs), args=args)
        
        docs_len = float(len(docs))

        ## WHY?
        for t in idf_dict.keys():
            idf_dict[t] =  np.log(docs_len / idf_dict[t] )
            # idf_dict[t] =  idf_dict[t] / docs_len 


        if args.token_value == "tf-idf-stop":
            stop_words = set(stopwords.words('english') )
            for t in stop_words:
                if t in idf_dict: 
                    idf_dict[t] *= 0.001

            def hasNumbers(inputString):
                return any(char.isdigit() for char in inputString)
            inp = " ".join([k for k in idf_dict.keys() if not hasNumbers(k)])
            spacy_nlp = spacy.load('en_core_web_sm')
            inp_results = [(token.text, token.tag_) for token in spacy_nlp(inp[:1000000])]
            allowed_tags = ['VB','NN','JJ','RB']   # UH for "yes", "no", etc.
            ignored_words = ['was','were','be','is','are','am',"'s","'re"] + ['do','did','done','does'] # verb of no info
            for word, tag in inp_results:
                if word in idf_dict.keys():
                    if len(tag)>=2 and tag[:2] in allowed_tags and (word not in ignored_words):
                        if tag[:2] in ['VB','NN']:
                            idf_dict[word] *= 4
                        else:
                            idf_dict[word] *= 2

        elif args.token_value == "df-stop":
            stop_words = set(stopwords.words('english') ) | set(['[SEP]', '[PAD]', '[CLS]', 'the', 'of', 'and', 'in', 'a', 'to', 'was', 'is', '"', 'for', 'on', 'as', 'with', 'by', 'he', "'s", 'at', 'that', 'from', 'it', 'his', 'an', 'which', 's', '.', ',', '(', ')',"'", '%'])  
            for k in idf_dict.keys():
                idf_dict[k] = 1.0/(idf_dict[k] + 1e-5)
            
            for t in stop_words:
                if t in idf_dict: 
                    idf_dict[t] = 0.01/(idf_dict[t]) 
  
            def hasNumbers(inputString):
                return any(char.isdigit() for char in inputString)
            inp = " ".join([k for k in idf_dict.keys() if not hasNumbers(k)])
            spacy_nlp = spacy.load('en_core_web_sm')
            inp_results = [(token.text, token.tag_) for token in spacy_nlp(inp[:1000000])]
            allowed_tags = ['VB','NN','JJ','RB']   # UH for "yes", "no", etc.
            ignored_words = ['was','were','be','is','are','am',"'s","'re"] + ['do','did','done','does'] # verb of no info
            for word, tag in inp_results:
                if word in idf_dict.keys():
                    if len(tag)>=2 and tag[:2] in allowed_tags and (word not in ignored_words):
                        if tag[:2] in ['VB','NN']:
                            idf_dict[word] *= 4
                        else:
                            idf_dict[word] *= 2
        

        logger.info("Calculate idf score finished")
        token_value = args.token_value
        

        if args.num_workers > 1:
            idx_list = [i for i in partitionIndexes(len(docs), args.epochs_to_generate)]
            writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
            arguments = [(docs, vocab_list, args, idx, idx_list[idx][0], idx_list[idx][1], idf_dict) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_training_file, arguments)
        else:
            idx_list = [i for i in partitionIndexes(len(docs), args.epochs_to_generate)]
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                create_training_file(docs, vocab_list, args, epoch, idx_list[epoch][0], idx_list[epoch][1], idf_dict, token_value)

        # duplicate epochs
        for epoch_num in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = args.output_dir / "epoch_{}.json".format(epoch_num)
            metrics_file = args.output_dir / "epoch_{}_metrics.json".format(epoch_num)
            for f in  range(1, args.duplicate_epochs):
                epoch_filename_dup = args.output_dir / "epoch_{}.json".format(epoch_num + args.epochs_to_generate * f)
                metrics_file_dup = args.output_dir / "epoch_{}_metrics.json".format(epoch_num + args.epochs_to_generate * f)
                os.system("cp {} {}".format(epoch_filename, epoch_filename_dup))
                os.system("cp {} {}".format(metrics_file, metrics_file_dup))


if __name__ == '__main__':
    main()
