# POINTER
This repository contains the implementation of the paper: "[**POINTER: Progressive Text Generation via Insertion-based Generative Pre-training**](https://arxiv.org/abs/2005.00558)"

![Screenshot](pointer.png) Figure: Illustration of the generation process (blue arrow) of the proposed POINTER model. At each stage, the module generates either a
or a special `NOI` token
for each gap between two existing tokens . The gener- ation stops when all the gaps predict `NOI`. The data preparation process (orange arrow) reverses the above gen- erative process.


## Live demo
The live demo can be found at [here](http://52.247.25.3:8900). Please expect delay and crash as it is running on a single GPU machine. 


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
