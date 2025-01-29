# GenBEE
The source code for the DASFAA 2025 paper: A Structure-aware Generative Model for Biomedical Event Extraction

## Environment

1. Please install the following packages from both conda and pip.

```
conda install
  - python 3.8
  - pytorch 2.0.1
  - numpy 1.24.3
  - ipdb 0.13.13
  - tqdm 4.65.0
  - beautifulsoup4 4.11.1
  - lxml 4.9.1
  - jsonnet 0.20.0
  - stanza=1.5.0
```
```
pip install
  - transformers 4.30.0
  - sentencepiece 0.1.96
  - scipy 1.5.4
  - spacy 3.1.4
  - nltk 3.8.1
  - tensorboardX 2.6
  - keras-preprocessing 1.1.2
  - keras 2.4.3
  - dgl-cu111 0.6.1
  - amrlib 0.7.1
  - cached_property 1.5.2
  - typing-extensions 4.4.0
  - penman==1.2.2
```

## Data
We support `MLEE`, `GE11`, and `PHEE`.


## Running

### Training
```
./train_model.sh 
```

### Evaluation

```
# Evaluating End-to-End EE
python ./evaluate_end2end.py --task E2E --data [eval_data] --model [saved_model_folder]

```

## Acknowledgement
Our scripts are developed based on the  [TextEE](https://arxiv.org/abs/2311.09562) framework. We deeply appreciate the contribution from the authors of the paper.

## Citation
```bib
@misc{yuan2024structureawaregenerativemodelbiomedical,
      title={A Structure-aware Generative Model for Biomedical Event Extraction}, 
      author={Haohan Yuan and Siu Cheung Hui and Haopeng Zhang},
      year={2024},
      eprint={2408.06583},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.06583}, 
}
```
