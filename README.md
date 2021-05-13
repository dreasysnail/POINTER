# POINTER
This repository contains the implementation of the EMNLP 2020 paper: "[**POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training**](https://arxiv.org/abs/2005.00558)", a progressive and non-autoregressive text generation pre-training approach. POINTER generates fluent text in a progressive and parallel manner. With empirical logarithmic time, POINTER outperforms existing non-autoregressive text generation approaches in hard-constrained text generation.

![Screenshot](pointer.png) Figure: Illustration of the generation process (blue arrow) of the proposed POINTER model. At each stage, the module generates either a
or a special `NOI` token
for each gap between two existing tokens . The generation stops when all the gaps predict `NOI`. The data preparation process (orange arrow) reverses the above generative process.

![Screenshot](example.png) Figure: Example of the progressive generation process


## News 
**\[Major Update 03/29/2021\]** The inference function and live demo now support phrases and short sentences as lexical constraints. When performing [inference](https://github.com/dreasysnail/POINTER#model-decoding), an additional "--sep" command can be added to specific the a user-specific separating token such as ";", to identify the boundries of the constraints. 


## LIVE DEMO
The live demo can be found at [here](http://52.247.25.3:8900). Please expect delay and crash as it is running on a single GPU machine. 



#### Setup Conda Environment

Please use the below commandlines to clone, install the requirements and load the Conda environment (Note that Cuda 10 is required):


```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```
We provide 3 ways to setup the python environments:
1. (Recommend) you can directly install the packages by running

```bash
bash env_setup.sh
```
2. or you can copy the exact same environment setup from us

```bash
conda create --prefix /pointer_env --file pointer-spec-file.txt
```
3. or using yaml file to reproduce the environment

```bash
conda env create -f pointer_env.yml -n pointer_env
conda activate pointer_env
```



#### Docker environment

To start, first install the docker and Nvidia-docker from their official repos. The image environment for running the code can be loaded as below:

Nvidia-docker v2.*

```bash
docker run --gpus all --ipc=host --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:ins_v4 bash
```

Nvidia-docker v1.*
```bash
$ nvidia-docker --rm -it -v $PWD:/workspace --network=host icaruszyz/large-scale-training:ins_v4 bash
```

## Rawdata 


Link to the data files can be downloaded as follows

| Dataset         |  Download link
|----------------------|--------|
| News| [\[link\]](https://yizzhang.blob.core.windows.net/insertiont/data/news/news_data.txt?sv=2019-10-10&st=2020-12-02T18%3A27%3A04Z&se=2030-12-03T18%3A27%3A00Z&sr=b&sp=r&sig=XfWYtYF8W8fFgbkJM1mrdLpb%2FoRJwQu5DTSaWrYv5yY%3D)  | 
| Restaurant review| [\[link\]](https://yizzhang.blob.core.windows.net/insertiont/data/yelp/yelp_160000.txt?sv=2019-12-12&st=2021-02-18T21%3A09%3A43Z&se=2028-06-19T20%3A09%3A00Z&sr=b&sp=r&sig=BvDirR%2FWgReKvfu6wd3qHM99rJUJ4KErEWhA4QBH5aE%3D)  | 


## POINTER model checkpoints 

Link to the model and config files can be downloaded as follows (345M models)

| Model           |  Download link
|----------------------|--------|
| Wiki pretrained model| [\[link\]](https://yizzhang.blob.core.windows.net/insertiont/ckpt/wiki.tar.gz?sv=2019-10-10&st=2021-03-10T21%3A40%3A57Z&se=2030-03-11T20%3A40%3A00Z&sr=b&sp=r&sig=oYI%2BKrj5wqeFV5jAF6EY15P8%2BtNGI%2F7FIOEox08QFDY%3D)  | 
| Restaurant review fine-tuned model| [\[link\]](https://yizzhang.blob.core.windows.net/insertiont/ckpt/yelp.tar.gz?sv=2019-10-10&st=2021-03-10T21%3A42%3A52Z&se=2030-03-11T20%3A42%3A00Z&sr=b&sp=r&sig=AEs%2F9hdWCde1D1RpPY9HVxc8ONBm%2B1P%2BhlbFSyPT%2Frc%3D)  | 
| News fine-tuned model| [\[link\]](https://yizzhang.blob.core.windows.net/insertiont/ckpt/news.tar.gz?sv=2019-10-10&st=2021-03-10T21%3A40%3A19Z&se=2030-03-11T20%3A40%3A00Z&sr=b&sp=r&sig=EPJ6WRchq0b3oEvrwlS0dnxFgJ5wjpgaJRl9kCUKSs4%3D)  | 

To continue, please decompress the file and move the `ckpt` folder into the main directory of this repo
```bash
tar -xzvf ckpt.tar.gz
```

  
  
## Generate from POINTER model with your own input


**Quick start (TL;DR)**:
Run the demo in our repo as
```bash
./demo.sh
```

**Decoding script**:
Please put an `test.key.txt` (see the `input/test.key.txt` in this repo for an example) into the `input` folder of this code, with `\t` seperating the constraints. The generation can be done using following command:
  
```bash
conda activate pointer_env
python inference.py \
--keyfile ./input/test.key.txt  \
--bert_model $model_path \
--output_dir $result_path \
```
The generation will be at the `$result_path` folder.



## Data preparation


**Data generation**:
```bash
python ./generate_training_data.py \
--train_corpus ./data/training.dummy.txt \
--bert_model bert-base-uncased \
--output_dir ./data/yelp_processed/ \
--clean  \
--task_name yelp
```


## Model training


**Dependency requirement**:

Please run `bash ./requirement.sh` to install the dependency required.
The `bert-large-uncased` model can be found at [here](https://huggingface.co/bert-large-uncased).
Please also install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.

**Pre-training**:
Below is an example of pretraining a model on wikipedia

```bash
python -m torch.distributed.launch  --nproc_per_node 16 training.py \
--pregenerated_data ./data/wikipedia_processed \
--bert_model bert-large-uncased \
--output_dir $WIKI_MODEL \
--epochs 40  \
--train_batch_size 64 \
--output_step 100000 \
--learning_rate 1e-5 
```

**Fine-tuning**:
Below is an example of finetuning a model with pretraining model ($WIKI_MODEL) [here](https://yizzhang.blob.core.windows.net/insertiont/ckpt.tar.gz?st=2020-08-18T20%3A49%3A02Z&se=2024-01-16T20%3A49%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=PKrSJt38cmY0P%2FBcZuyK%2Btm3bXyYzzfazaqTu1%2F%2FDtc%3D)

```bash
python -m torch.distributed.launch  --nproc_per_node 16 training.py \
--pregenerated_data ./data/yelp_processed \
--bert_model $WIKI_MODEL \
--output_dir $finetune_model_path \
--epochs 40 \
--train_batch_size 64 \
--output_step 100000 \
--learning_rate 1e-5\
```

## Model decoding


**Keywords extraction**:
First, you can use the following script to generate a bunch of keywords for the test file. 
```bash
python keyword_extraction.py \
--n_keys 7 \
--file ./data/yelp_test.txt
```
The keywords file will be saved in the same folder as the input file (in this example, the keywords file will be `./data/yelp_test.key.txt`)

**Generating from the keywords**
With the trained model, you can generate a sentence from a give keywords file. The keywords file can be obtained by the previous keywords extraction step, or by a custom user input file. The following commands show an example of how to decode from the keywords file generated from last step:
```bash
python inference.py \
--keyfile ./data/yelp_test.key.txt  \
--bert_model $finetune_model_path \
--output_dir $result_path 
```

The inference function and live demo now support phrases and short sentences as lexical constraints. When performing [inference](https://github.com/dreasysnail/POINTER#model-decoding), an additional "--sep" command can be added to specific the a user-specific separating token such as ";", to identify the boundries of the constraints. The default separator is white space " ". 

NOTE THAT, if using default white space separator, the input keywords file will be tokenized by a BERT tokenizer. A less common word will likely be parsed into subwords, for example, `cheesecake` will be split into `cheese`, `##cake`. As a result the final generation may not contain the whole word `cheesecake`. 


## Citation
If you use this code in your research, you can cite our [paper](https://arxiv.org/abs/2005.00558):
```bash
@inproceedings{zhang2020pointer,
  title={POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training},
  author={Zhang, Yizhe and Wang, Guoyin and Li, Chunyuan and Gan, Zhe and Brockett, Chris and Dolan, Bill},
  booktitle={EMNLP},
  year={2020}
}
```
