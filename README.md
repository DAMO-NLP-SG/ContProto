# ContProto

This repository contains the code for the ACL 2023 paper "Improving Self-training for Cross-lingual Named Entity Recognition with Contrastive and Prototype Learning"
![alt text](https://github.com/DAMO-NLP-SG/ContProto/blob/main/contproto.png?raw=true)

## Requirements
* torch>=1.6
* pytorch-lightning>=0.9.0
* tokenizers
* transformers

To install the dependencies, run:
```
pip install -r requirements.txt
```
Our experiments are run on a single Nvidia V100 32GB GPU.

## Data Format
Before running the experiments, you will need to convert the NER dataset from CoNLL format to span-based format by running the following scripts:
```
cd data
bash data_preprocess.sh
```
We have provided the processed German dataset from CoNLL as an example in `data/conll03_de`.

## Generating Pseudo-labeled Data
To generate the pseudo-labeled data, run the following script:
```
bash run_generate_pseudo.sh de conll03
```
For example, the generated German pseudo labels will be stored in `train_logs/conll03/genpseudo_de_0/`.

## Training
To start training of ContProto, run the following script:
```
bash run_contpro_auto-margin.sh de conll03
```

## Citation
If you find this repository useful, please cite our paper:
```
@inproceedings{zhou-etal-2023-improving,
    title = "Improving Self-training for Cross-lingual Named Entity Recognition with Contrastive and Prototype Learning",
    author = "Zhou, Ran  and
      Li, Xin  and
      Bing, Lidong  and
      Cambria, Erik  and
      Miao, Chunyan",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023",
}
```
