# Self-Attention Network for Natural Language Inference
Pytorch re-implementation of [Distance-based Self-Attention Network for Natural Language Inference](https://arxiv.org/abs/1712.02047) without distance mask.
This is an unofficial implementation.

## Results
Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model | Valid Acc(%) | Test Acc(%)
| ----- | ------------ | -----------
| Baseline from the paper (without distance mask) | - | 86.0 |
| Re-implemenation | 86.2 | 85.7 |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.6
- Pytorch: 0.4.0

## Requirements
Please install the following library requirements first.

    nltk==3.3
    tensorboardX==1.2
    torch==0.4.0
    torchtext==0.2.3
    
## Training
> python train.py --help

    usage: train.py [-h] [--batch-size BATCH_SIZE] [--data-type DATA_TYPE]
                [--dropout DROPOUT] [--epoch EPOCH] [--gpu GPU]
                [--learning-rate LEARNING_RATE] [--print-freq PRINT_FREQ]
                [--word-dim WORD_DIM] [--num-heads NUM_HEADS] [--d-ff D_FF]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --data-type DATA_TYPE
      --dropout DROPOUT
      --epoch EPOCH
      --gpu GPU
      --learning-rate LEARNING_RATE
      --print-freq PRINT_FREQ
      --word-dim WORD_DIM
      --num-heads NUM_HEADS
      --d-ff D_FF

 
 **Note:** 
- Only codes to use SNLI as training data are implemented.
- The Dropout and Layer Normalization technique exist in this model, but it is not clear how those are applied by the paper.
