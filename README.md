# Fine-Tuned Masked Language Model

This repository contains a fine-tuned version of the `distilbert-base-uncased` model, optimized for masked language modeling tasks. The fine-tuning process has resulted in a model that performs effectively on an imdb dataset with specific evaluation metrics detailed below.

<img width="526" alt="Screenshot 2024-05-25 at 7 28 28 PM" src="https://github.com/cxx5208/Finetuned-MLM-accelerate/assets/76988460/fd53f4b0-cfb5-4866-a8e7-2222618698b9">

## Datacard: https://huggingface.co/datasets/stanfordnlp/imdb
## Model Performance

The model achieves the following results on the evaluation set:

- **Loss:** 2.4252

## Model Description

This model builds on `distilbert-base-uncased`, a smaller, faster, cheaper, and lighter version of BERT, while retaining 97% of BERT’s language understanding. It has been fine-tuned to predict masked tokens in sentences, making it suitable for various natural language processing tasks that require contextual understanding.

## Intended Uses & Limitations

- **Intended Uses:** The model can be used for any task requiring a deep understanding of language context, such as text completion, data augmentation, and other NLP applications.
- **Limitations:** Specific details about the dataset and potential biases are not provided, and the model’s performance is only validated on the evaluation set. Use caution when applying it to significantly different contexts or domains.

## Training Procedure

The model was trained with the following hyperparameters:

- **Learning Rate:** 2e-05
- **Train Batch Size:** 64
- **Eval Batch Size:** 64
- **Seed:** 42
- **Optimizer:** Adam with `betas=(0.9, 0.999)` and `epsilon=1e-08`
- **Learning Rate Scheduler Type:** Linear
- **Number of Epochs:** 3.0
- **Mixed Precision Training:** Native AMP

### Training Results

| Training Loss | Epoch | Step | Validation Loss |
|---------------|-------|------|-----------------|
| 2.3173        | 1.0   | 157  | 2.3981          |
| 2.3598        | 2.0   | 314  | 2.3818          |
| 2.3838        | 3.0   | 471  | 2.4175          |

## Framework Versions

- **Transformers:** 4.41.1
- **Pytorch:** 2.3.0+cu121
- **Datasets:** 2.19.1
- **Tokenizers:** 0.19.1


