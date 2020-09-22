# POINTER
This repository contains the implementation of the paper: "[**POINTER: Constrained Text Generation via Insertion-based Generative Pre-training**](https://arxiv.org/abs/2005.00558)", a progressive and non-autoregressive text generation pre-training approach. 

![Screenshot](pointer.png) Figure: Illustration of the generation process (blue arrow) of the proposed POINTER model. At each stage, the module generates either a
or a special `NOI` token
for each gap between two existing tokens . The generation stops when all the gaps predict `NOI`. The data preparation process (orange arrow) reverses the above generative process.


## Live demo
The live demo can be found at [here](http://52.247.25.3:8900). Please expect delay and crash as it is running on a single GPU machine. 


## Generating from pretrained POINTER models


#### Setup Conda Environment

Please use the below commandlines to clone, install the requirements and load the Conda environment (Note that Cuda 10 is required):


```bash
sudo apt-get install -y make wget gzip bzip2 xz-utils zstd
```

```bash
conda env create -f pointer_env.yml -n pointer_env
conda activate pointer_env
```

If you run this on an architecture other than Linux, please use `LSP-generic.yml` instead of `LSP-linux.yml` but please note that the generic one is not tested in all platform, so the stablity can not be gauranteed.
  
#### 

Link to the model and config files can be downloaded [here](https://yizzhang.blob.core.windows.net/insertiont/ckpt.tar.gz?st=2020-08-18T20%3A49%3A02Z&se=2024-01-16T20%3A49%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=PKrSJt38cmY0P%2FBcZuyK%2Btm3bXyYzzfazaqTu1%2F%2FDtc%3D).  

To continue, please decompress the file and move the `ckpt` folder into the main directory of this repo
```bash
tar -xzvf ckpt.tar.gz
```

  
  
#### Generate from INSET model with your own input


**Quick start (TL;DR)**:
Run the demo in our repo as
```bash
./demo.sh
```

**Decoding script**:
Please put an `test.key.txt` (see the `input/test.key.txt` in this repo for an example) into the `input` folder of this code, with `\t` seperating the constraints. The generation can be done using following command:
  
```bash
conda activate pointer_env
python ./code/inference.py \
--pregenerated_data ./input  \
--bert_model $model_path \
--output_dir $result_path \
```
The generation will be at the `$result_path` folder.

## Citation
If you use this code in your research, you can cite our [paper](https://arxiv.org/abs/2005.00558):
```bash
@article{zhang2020pointer,
  title={POINTER: Constrained Text Generation via Insertion-based Generative Pre-training},
  author={Zhang, Yizhe and Wang, Guoyin and Li, Chunyuan and Gan, Zhe and Brockett, Chris and Dolan, Bill},
  journal={arXiv preprint arXiv:2005.00558},
  year={2020}
}
```
