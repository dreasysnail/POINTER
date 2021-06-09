from argparse import ArgumentParser
from pathlib import Path
import os
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory


from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertForMaskedLM
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch.nn.functional as F
from util import MAX_TURN, PREVENT_FACTOR, PROMOTE_FACTOR, PREVENT_LIST, REDUCE_LIST, STOP_LIST, boolean_string


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids no_ins")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)
NOI_ID = 1
class Node(object):
    def __init__(self, input_ids, segment_ids, input_mask, score, shift, length, pos_start, input_len_start):
        super(Node, self).__init__()
        self.input_ids = input_ids
        self.segment_ids = segment_ids  # parent Node, None for root
        self.input_mask = input_mask
        self.score = score
        self.shift = shift
        self.length=length
        self.pos_start=pos_start
        self.input_len_start=input_len_start

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def convert_example_to_features(example, tokenizer, max_seq_length, id = 0, no_ins_at_first = False, tokenizing = False):
    tokens = ["[CLS]"] + example  
    if len([x for t in tokens for x in tokenizer.encode(t)]) > max_seq_length:
        logging.info(f"Warning: input id-{id} exceeds max sequence length limit!")
        tokens = ["[CLS]"] + ["Error : Input exceeds length limit;"]

    no_ins = [0] if no_ins_at_first else []
    if tokenizing:
        #input_ids = tokenizer.encode(" ".join(tokens))
        input_ids = [x for t in tokens for x in tokenizer.encode(t)]
        input_ids_lens = [len(tokenizer.encode(t)) for t in tokens]
        cur = 0
        for l in input_ids_lens:
            if l >=2 :
                no_ins.extend([cur + x for x in range(0,l-1)])
            cur += l
    else:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)

    no_ins_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)

    no_ins_array[:len(no_ins)] = no_ins

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             no_ins=no_ins_array,
                             )
    return features

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, max_seq_len = 256, sep=" ", no_ins_at_first = False, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path
        num_samples = sum(1 for line in open(data_file))
        self.num_samples = num_samples
        seq_len = max_seq_len
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            no_ins = np.memmap(filename=self.working_dir/'no_ins.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            no_ins =  np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            
        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                if i >= num_samples:
                    break
                line = line.strip()
                example = [s.lstrip().strip() for s in line.split(sep)]
                features = convert_example_to_features(example, tokenizer, seq_len, no_ins_at_first = no_ins_at_first, id = i, tokenizing=True)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                no_ins[i] = features.no_ins
        if i != num_samples - 1:
            logging.info("i={} not equal to num_samples={}".format(i, num_samples))
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.no_ins = no_ins


    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.no_ins[item].astype(np.int64)),
                )



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def greedy_search(model, input_ids, segment_ids, input_mask, no_ins = None, device='cuda', temperature=1.0, args=None, tokenizer=None, prevent=None, promote=None, reduce=None, verbose = None):
    # print("greedy generation")
    if not verbose:
        verbose = args.verbose
    zero_list = ["[", "]", "(", ")"]
    zero_ids = [ tokenizer.vocab.get(x) for x in zero_list]
    if verbose >0:
        print("\nInput %s" % (" ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in input_ids[0].detach().cpu().numpy() if x!=0])))

    no_ins_cur = no_ins[0][:(no_ins[0]==-1).nonzero()[0]]
    for ip in range(MAX_TURN):
        with torch.no_grad():
            result= model(input_ids, segment_ids, input_mask)
            mask_prediction_scores = result[0]
            input_len = torch.sum(input_mask,1)

            noi_temp = min(float(ip) / args.noi_decay, 1.0) 
            mask_prediction_scores[:,:,1] = mask_prediction_scores[:,:,1] * noi_temp
            logits = mask_prediction_scores / temperature

            if prevent:
                for p in prevent:
                    logits[:,:,p] = logits[:,:,p] * PREVENT_FACTOR
            if reduce:
                reduce_factor = min(float(ip) / args.reduce_decay, 1.0) 
                for p in reduce:
                    logits[:,:,p] = logits[:,:,p] * reduce_factor
            if promote:
                for p in promote:
                    logits[:,:,p] = logits[:,:,p] * PROMOTE_FACTOR 
            if args.lessrepeat:
                for p in input_ids.cpu().numpy()[0]:
                    logits[:,:,p] = logits[:,:,p] * 0.8            

            logits[:,:, zero_ids] = -1e10

            probs = F.softmax(logits, dim=-1)

            input_ids_new = torch.zeros_like(input_ids)
            top_predicts = torch.zeros([input_ids.shape[0], input_ids.shape[1], 3], dtype=torch.long)

            mask_predicts = probs.argmax(2)
            for t in range(args.max_seq_length):
                top_predicts[:,t] = torch.topk(probs[:,t,:], k=3)[1]


            input_mask_new = torch.zeros_like(input_mask)
            logit_new = torch.zeros_like(input_ids,dtype=torch.float)
            input_ids_ori = input_ids
            top_predicts_new = torch.zeros_like(top_predicts)
            i = 0
            j = 0
            k = 0
            sep_tok = tokenizer.vocab['[SEP]']
            # update no_ins
            mask_predicts[0][no_ins_cur] = NOI_ID #
            new_no_ins_cur = no_ins_cur.clone().detach()
            # [tokenizer.decode([x.tolist()]) for x in input_ids[0] if x!= 0]
            while np.max([i,j,k]) < args.max_seq_length-1:
                input_ids_new[0,k] = input_ids[0,i]
                if input_ids[0,i] == 0: # padding, ignore prediction
                    break
                if input_ids[0,i] == sep_tok:
                    break
                i += 1
                k += 1

                if mask_predicts[0,j].cpu().numpy() != NOI_ID:
                    input_ids_new[0,k] = mask_predicts[0,j]
                    logit_new[0,k] = probs[0,j,mask_predicts[0,j]]
                    top_predicts_new[0,k,:] = top_predicts[0,j,:]  
                    if len(no_ins_cur)> 0 and no_ins_cur[-1] > j:
                        new_no_ins_cur[torch.where(no_ins_cur > j)[0][0]:] += 1
                    k+=1
                    j+=1
                    

                else:
                    j+=1

            no_ins_cur = new_no_ins_cur    
            mask_pos = input_ids_new > 1
            input_ids = input_ids_new
            input_mask = mask_pos

            logit_new = logit_new.detach().cpu().numpy()
            top_predicts_new = top_predicts_new.detach().cpu().numpy()
            if verbose == 0:
                pass
            elif verbose == 2:
                print("Round %d: %s" % (ip, " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) + (("(" + "{:.2f}".format(float(logit_new[0,i])) + ")") if logit_new[0,i] > 0 else "")  for i, x in enumerate(input_ids[0].detach().cpu().numpy()) if x!=0])))
            elif verbose == 3:
                print("Round %d: %s" % (ip, " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) + (("(" + "{:.2f}".format(float(logit_new[0,i])) + " "+ " ".join([str(tokenizer.ids_to_tokens.get(y, "noa").encode('ascii', 'ignore').decode('ascii')) for y in top_predicts_new[0,i,:]]) + ")") if logit_new[0,i] > 0 else "")  for i, x in enumerate(input_ids[0].detach().cpu().numpy()) if x!=0])))
            else:
                print("Round %d: %s" % (ip, " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in input_ids[0].detach().cpu().numpy() if x!=0])))
    return input_ids

def sample_generate(model, input_ids, segment_ids, input_mask, no_ins = None, device='cuda', temperature=1.0, args=None, tokenizer=None, sample_num=1, top_k=10, top_p=0.9, prevent=None, promote=None, reduce=None, verbose = None):
    if not verbose:
        verbose = args.verbose
    zero_list = ["[", "]", "(", ")"]
    zero_ids = [ tokenizer.vocab.get(x) for x in zero_list]
    if verbose>0:
        print("\nInput %s" % (" ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in input_ids[0].detach().cpu().numpy() if x!=0])))
    no_ins_cur = no_ins[0][:(no_ins[0]==-1).nonzero()[0]]
    for ip in range(MAX_TURN):
        with torch.no_grad():
            result= model(input_ids, segment_ids, input_mask)
            mask_prediction_scores = result[0]
            input_len = torch.sum(input_mask,1)

            noi_temp = min(float(ip) / args.noi_decay, 1.0) 
            mask_prediction_scores[:,:,1] = mask_prediction_scores[:,:,1] * noi_temp
            logits = mask_prediction_scores / temperature

            if prevent:
                for p in prevent:
                    logits[:,:,p] = logits[:,:,p] * PREVENT_FACTOR 
            if reduce:
                reduce_factor = min(float(ip) / args.reduce_decay, 1.0) 
                for p in reduce:
                    logits[:,:,p] = logits[:,:,p] * reduce_factor
            if promote:
                for p in promote:
                    logits[:,:,p] = logits[:,:,p] * PROMOTE_FACTOR 
            if args.lessrepeat:
                for p in input_ids.cpu().numpy()[0]:
                    logits[:,:,p] = logits[:,:,p] * 0.8
            
            

            logits[:,:, zero_ids] = -1e10
            for i in range(args.max_seq_length):
                logits[:,i] = top_k_top_p_filtering(logits[:,i].squeeze(), top_k = top_k, top_p = top_p)
            probs = F.softmax(logits, dim=-1)

            input_ids_new = torch.zeros_like(input_ids)
            top_predicts = torch.zeros([input_ids.shape[0], input_ids.shape[1], 3], dtype=torch.long)
            mask_predicts = torch.zeros_like(input_ids, dtype=torch.long)
            for t in range(args.max_seq_length):
                mask_predicts[:,t] =torch.multinomial(probs[:,t,:], num_samples=1)
                top_predicts[:,t] = torch.topk(probs[:,t,:], k=3)[1]


            logit_new = torch.zeros_like(input_ids,dtype=torch.float)
            input_ids_ori = input_ids
            top_predicts_new = torch.zeros_like(top_predicts)
            i = 0
            j = 0
            k = 0
            sep_tok = tokenizer.vocab['[SEP]']
            # update no_ins
            mask_predicts[0][no_ins_cur] = NOI_ID #
            new_no_ins_cur = no_ins_cur.clone().detach()
            while np.max([i,j,k]) < args.max_seq_length-1:
                # print(i,j,k)
                input_ids_new[0,k] = input_ids[0,i]
                if input_ids[0,i] == 0: # padding, ignore prediction
                    break
                if input_ids[0,i] == sep_tok:
                    break
                
                i += 1
                k += 1

                if mask_predicts[0,j].cpu().numpy() != 1:
                    input_ids_new[0,k] = mask_predicts[0,j]
                    logit_new[0,k] = probs[0,j,mask_predicts[0,j]]
                    top_predicts_new[0,k,:] = top_predicts[0,j,:]   
                    if len(no_ins_cur)> 0 and no_ins_cur[-1] > j:
                        new_no_ins_cur[torch.where(no_ins_cur > j)[0][0]:] += 1                 
                    k+=1
                    j+=1
                else:
                    j+=1
            
            no_ins_cur = new_no_ins_cur 
            mask_pos = input_ids_new > 1
            input_ids = input_ids_new
            input_mask = mask_pos
            

            logit_new = logit_new.detach().cpu().numpy()
            top_predicts_new = top_predicts_new.detach().cpu().numpy()
            if verbose == 0:
                pass
            elif verbose == 2:
                print("Round %d: %s" % (ip, " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) + (("(" + "{:.2f}".format(float(logit_new[0,i])) + ")") if logit_new[0,i] > 0 else "")  for i, x in enumerate(input_ids[0].detach().cpu().numpy()) if x!=0])))
            elif verbose == 3:
                print("Round %d: %s" % (ip, " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) + (("(" + "{:.2f}".format(float(logit_new[0,i])) + " "+ " ".join([str(tokenizer.ids_to_tokens.get(y, "noa").encode('ascii', 'ignore').decode('ascii')) for y in top_predicts_new[0,i,:]]) + ")") if logit_new[0,i] > 0 else "")  for i, x in enumerate(input_ids[0].detach().cpu().numpy()) if x!=0])))
            else:
                print("Round %d: %s" % (ip, " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in input_ids[0].detach().cpu().numpy() if x!=0])))
    return input_ids
    



def main():
    parser = ArgumentParser()
    parser.add_argument('--keyfile', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=False, default=None)
    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", 
                        type=boolean_string, 
                        default=False, 
                        )
    parser.add_argument("--reduce_memory",                        
                        type=boolean_string, 
                        default=False, 
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", 
                        type=boolean_string, 
                        default=False, 
                        help="Whether not to use CUDA when available")
    parser.add_argument("--batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16', 
                        type=boolean_string, 
                        default=False, 
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--type",
                        default="greedy",
                        type=str,
                        choices=['greedy','sampling'],
                        help="greedy: greedy generation. sampling: top-k sampling")
    parser.add_argument('--noi_decay',
                        type=int,
                        default=1,
                        help="round number to decay NOI prob") 
    parser.add_argument('--reduce_decay',
                        type=int,
                        default=1,
                        help="round number to decay reduce prob") 
    parser.add_argument('--verbose', type=int,
                        default=0,
                        help="verbose level") 
    parser.add_argument('--n_test',
                        type=int,
                        default=5000,
                        help="number of test examples")
    parser.add_argument('--prevent', 
                        type=boolean_string, 
                        default=True,
                        help="avoid generating several words")
    parser.add_argument('--reduce_stop',
                        type=boolean_string, 
                        default=True, 
                        help="reduce stopwords")    
    parser.add_argument('--lessrepeat',
                        type=boolean_string, 
                        default=True, 
                        help="reduce repetition (only for tokenwise)")
    parser.add_argument('--sep',
                         type=str, default=" ", help="token to seperate keywords")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=256,
                        help="max sequence length") 
    parser.add_argument("--no_ins_at_first", 
                        type=boolean_string, 
                        default=False, 
                        help="Do not insert at the begining of the text")

                        
    args = parser.parse_args()



    if not args.output_dir:
        args.output_dir = args.bert_model

    epoch_file = args.keyfile
    # args.max_seq_length = 256
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)

    args.output_mode = "classification"


    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)


    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model)

    sep_tok = tokenizer.vocab['[SEP]']
    cls_tok = tokenizer.vocab['[CLS]']
    pad_tok = tokenizer.vocab['[PAD]']

    model.to(device)
    model.eval()

    print(args)    

    epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.keyfile, tokenizer=tokenizer, max_seq_len = args.max_seq_length, sep=args.sep, no_ins_at_first = args.no_ins_at_first, num_data_epochs=1)
    epoch_sampler = SequentialSampler(epoch_dataset)
    generate_dataloader = DataLoader(epoch_dataset, sampler=epoch_sampler,batch_size=args.batch_size)
    file_name = os.path.join(args.output_dir, os.path.basename(args.keyfile)[:-3] + os.path.basename(args.bert_model) + f".{args.type}.txt")
    f = open(file_name, "w", 1)


    logging.info("***** Running generation *****")
    logging.info(f"  Num examples = {epoch_dataset.num_samples}")
    logging.info("  Batch size = %d", args.batch_size)
    logging.info(f"  Save to {file_name}")


    prevent = [ tokenizer.vocab.get(x) for x in PREVENT_LIST] if args.prevent else None
    if args.reduce_stop:
        # import pdb; pdb.set_trace()
        reduce_l = REDUCE_LIST |  STOP_LIST
    reduce = None
    if args.prevent:
        reduce = [ tokenizer.vocab.get(x) for x in reduce_l]  
        reduce = [s for s in reduce if s]


    with tqdm(total=len(generate_dataloader), desc=f"Epoch {0}") as pbar:
        for step, batch in enumerate(generate_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, lm_label_ids, no_ins = batch
            if args.type == "greedy":
                predict_ids = greedy_search(model, input_ids, segment_ids, input_mask, no_ins = no_ins, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            elif args.type == 'sampling':
                predict_ids = sample_generate(model, input_ids, segment_ids, input_mask, no_ins = no_ins, temperature=0.8, args=args, tokenizer=tokenizer, prevent=prevent, reduce= reduce)
            else:
                raise NotImplementedError
            output =  " ".join([str(tokenizer.ids_to_tokens.get(x, "noa").encode('ascii', 'ignore').decode('ascii')) for x in predict_ids[0].detach().cpu().numpy() if x!=sep_tok and x != pad_tok and x != cls_tok]) + "\n" 
            output = output.replace(" ##", "")
            f.write(output)
            pbar.update(1)
            


if __name__ == '__main__':
    main()
