Code for our ACL 2021 paper - [Modeling Discriminative Representations for Out-of-Domain Detection with Supervised Contrastive Learning](https://arxiv.org/pdf/2105.14289.pdf)

# Requirements

```
python 3.6.2
torch 1.4.0
tensorflow 1.14.0
```
For detailed dependencies, please refer to requirements.txt

# Get Started

prepare Glove or BERT pretrained embeddings from https://github.com/stanfordnlp/GloVe and https://github.com/google-research/bert

put the embedding file glove.6B.300d.txt into ./glove_embeddings

modify the script to train or test model in different modes

```
bash run.sh
```

# Citation

```
@inproceedings{Zeng2021ModelingDR,
  title={Modeling Discriminative Representations for Out-of-Domain Detection with Supervised Contrastive Learning},
  author={Zhiyuan Zeng and Keqing He and Yuanmeng Yan and Zijun Liu and Yanan Wu and Hong Xu and Huixing Jiang and Weiran Xu},
  booktitle={ACL/IJCNLP},
  year={2021}
}
```
