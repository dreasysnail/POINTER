# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import numpy as np
import pdb
from collections import Counter
from nltk.corpus import stopwords
import spacy
# import yake
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        # self.lm_label_ids = lm_label_ids


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    # def get_labels(self):
    #     """Gets the list of labels for this data set."""
    #     raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class CocoProcessor(DataProcessor):
    """Processor for the Coco data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_labels(self, tokenizer):
        """See base class."""
        return list(tokenizer.vocab.keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a))
        return examples


class NewsProcessor(DataProcessor):
    """Processor for the News data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_labels(self, tokenizer):
        """See base class."""
        return list(tokenizer.vocab.keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a))
        return examples

class WikiProcessor(DataProcessor):
    """Processor for the Wiki data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "en.test")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "en.valid")), "dev")

    def get_labels(self, tokenizer):
        """See base class."""
        return list(tokenizer.vocab.keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) > 0:
                text_a = line[0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a))
        return examples

class WikiProcessor_fast(DataProcessor):
    """Processor for the Wiki data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "wikidoc.tk")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "en.valid")), "dev")

    def get_labels(self, tokenizer):
        """See base class."""
        return list(tokenizer.vocab.keys())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if i % 1000000 == 0:
                logger.info("Loading example %d of %d" % (i, len(lines)))
            if len(line) > 0:
                text_a = line[0]
                if len(text_a.split()) > 1:
                    examples.append(
                        InputExample(guid=guid, text_a=text_a))
        return examples

def house_robber(prob_list, skip = 2):
    pos = [0] * len(prob_list)
    count = [0] * len(prob_list)
    pos[0] = []
    count[0] = 0.
    if len(prob_list) <= skip:
        return [np.argmax(prob_list) + 1]
    for s in range(1,skip):
        pos[s] = [s]
        count[s] = prob_list[s]

    for i in range(skip, len(prob_list)):
        # import pdb; pdb.set_trace()
        if prob_list[i] + count[i-skip] > max([count[i-j] for j in range(1,skip)]):
            pos[i] = pos[i-skip].copy()
            pos[i].append(i)
            count[i] = prob_list[i] + count[i-skip]
        else:
            max_id = np.argmax([count[i-j] for j in range(1,skip)]) + 1
            pos[i] = pos[i-max_id].copy()
            count[i] = count[i-max_id]
    return pos[-1]

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', noi_token='[NOI]', pad_token=0,
                                 sequence_a_segment_id=0,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, args=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    noi_token_id = tokenizer.convert_tokens_to_ids(noi_token)
    num_exm = len(examples)
    idf_dict = {}
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100000 == 0:
            logger.info("Writing idf example %d of %d" % (ex_index, len(examples)))
        if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'bert-large-uncased' :
            tokens_a = tokenizer.tokenize(example.text_a)
        elif args.model_name_or_path == 'bert-base-cased':
            tokens_a = example.text_a.split()
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        for t in tokens:
            idf_dict[t] = idf_dict.get(t, 0) + 1
    for t in idf_dict.keys():
        idf_dict[t] =  idf_dict[t] / num_exm
    

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)


        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        prob_list = np.array([idf_dict[t] for t in tokens])
        N = len(tokens)
        lm_label_ids = [noi_token_id] * max_seq_length

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              lm_label_ids=lm_label_ids))
        while N > 1:
            mask_pos = np.array(house_robber(prob_list))
            unmask_pos = np.setdiff1d(np.arange(N), mask_pos)

            tokens = [t  for i,t in enumerate(tokens) if i in unmask_pos]
            N = len(tokens)

            # mask_lm_label_ids = input_ids 
            lm_label_ids = [pad_token] * max_seq_length
            j=0
            i = 1
            while i < len(prob_list):
                if i in mask_pos:
                    lm_label_ids[j] = input_ids[i]  
                    i += 2
                else:
                    lm_label_ids[j] = noi_token_id
                    i += 1
                j += 1
            # print(i,j)
            while j < len(unmask_pos):
                lm_label_ids[j] = noi_token_id # no input for last token of new sequence
                j+= 1

            prob_list = prob_list[unmask_pos]
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(unmask_pos)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))
                # logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                lm_label_ids=lm_label_ids))

    
    return features


def convert_examples_to_features_yake(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', noi_token='[NOI]', pad_token=0,
                                 sequence_a_segment_id=0,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, args=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    noi_token_id = tokenizer.convert_tokens_to_ids(noi_token)
    num_exm = len(examples)
    idf_dict = {}
    for (ex_index, example) in enumerate(examples):
        if ex_index % 100000 == 0:
            logger.info("Writing idf example %d of %d" % (ex_index, len(examples)))
        if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'bert-large-uncased':
            tokens_a = tokenizer.tokenize(example.text_a)
        elif args.model_name_or_path == 'bert-base-cased':
            tokens_a = example.text_a.split()
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        for t in tokens:
            idf_dict[t] = idf_dict.get(t, 0) + 1
    for t in idf_dict.keys():
        idf_dict[t] =  idf_dict[t] / num_exm
    
    stop_words = set(stopwords.words('english') )

    for t in stop_words:
        if t in idf_dict: 
            idf_dict[t] *= 0.001

    inp = " ".join(idf_dict.keys())
    spacy_nlp = spacy.load('en_core_web_sm')
    inp_results = [(token.text, token.tag_) for token in spacy_nlp(inp)]
    allowed_tags = ['VB','NN','JJ','RB']   # UH for "yes", "no", etc.
    ignored_words = ['was','were','be','is','are','am',"'s","'re"] + ['do','did','done','does'] # verb of no info
    for word, tag in inp_results:
        if word in idf_dict.keys():
            if len(tag)>=2 and tag[:2] in allowed_tags and (word not in ignored_words):
                if tag[:2] in ['VB','NN']:
                    idf_dict[word] *= 4
                else:
                    idf_dict[word] *= 2


    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)


        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        tf = Counter(tokens)
        tokens_len = float(len(tokens))
        # score: higher will be more likely to be keeped
        prob_list =  np.array([idf_dict[t] * tf[t] / tokens_len  for t in tokens]) 
        # prob_list = np.array([idf_dict[t] for t in tokens])

        # add yake
        key_word_len = 100
        kw_extractor = yake.KeywordExtractor()
        keywords = kw_extractor.extract_keywords(" ".join(tokens))
        key_word_len = len(keywords)
        for i, t in enumerate(tokens):
            if  t in keywords:
                prob_list[i] *= 100  
        # Repeat words
        for i, t in enumerate(tokens):
            if t in tokens[:i]:
                prob_list[i] /= 10


            

        prob_list = max(prob_list) - prob_list

        N = len(tokens)
        lm_label_ids = [noi_token_id] * max_seq_length

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              lm_label_ids=lm_label_ids))
        while N > 1:
            mask_pos = np.array(house_robber(prob_list))
            unmask_pos = np.setdiff1d(np.arange(N), mask_pos)

            tokens = [t  for i,t in enumerate(tokens) if i in unmask_pos]
            N = len(tokens)

            # mask_lm_label_ids = input_ids 
            lm_label_ids = [pad_token] * max_seq_length
            j=0
            i = 1
            while i < len(prob_list):
                if i in mask_pos:
                    lm_label_ids[j] = input_ids[i]  
                    i += 2
                else:
                    lm_label_ids[j] = noi_token_id
                    i += 1
                j += 1
            # print(i,j)
            while j < len(unmask_pos):
                lm_label_ids[j] = noi_token_id # no input for last token of new sequence
                j+= 1

            prob_list = prob_list[unmask_pos]
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(unmask_pos)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("lm_label_ids: %s" % " ".join([str(x) for x in lm_label_ids]))


            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                lm_label_ids=lm_label_ids))

    
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "coco":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "news":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wiki":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "PS":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "yelp":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)

processors = {
    "news": NewsProcessor,
    "coco": CocoProcessor,
    "wiki": WikiProcessor_fast,
}




