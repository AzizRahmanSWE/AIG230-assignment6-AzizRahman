# AIG230 Assignment 6: Natural Language Processing
**By:** Aziz Rahman  

## Project Overview
This repository contains the implementation and analysis of two language models: a Statistical Trigram Language Model and a Recurrent Neural Network (RNN) Language Model. Both models were trained to predict the next token in a sequence using the NLTK Brown corpus (news category) to compare statistical versus neural approaches on a relatively small dataset.

## Repository Structure & The Notebooks
This repository includes two distinct Jupyter Notebooks to showcase the development process and the final structured deliverable:

### 1. `practice_code(optional)/AIG230_assignment6_AzizRahman.ipynb`
This is the initial practice scratchpad. It contains the raw, ground-up implementation of the assignment concepts, including:
* **Data Preparation (Part D):** The pipeline for loading, lowercasing, and tokenizing the Brown corpus, including adding `<bos>`, `<eos>`, and `<unk>` special tokens.
* **Statistical N-gram Model (Part A):** The implementation of the optional Trigram model using `nltk.lm` with Laplace (Add-1) smoothing, alongside perplexity evaluation and text generation.
* **Initial RNN Build:** The first draft of the PyTorch RNN architecture and training loop to test the mathematical logic before migrating to the structured template.

### 2. `start_code/assignment_part_B_starter.ipynb`
This is the official, final deliverable for the mandatory Part B of the assignment. It uses the structured, object-oriented approach provided via the professor's starter code. It contains:
* **Numericalization & Datasets:** Custom `NextTokenStreamDataset` classes and PyTorch `DataLoader` setups to create sequential input/target pairs.
* **RNN Architecture:** A custom `RNNLanguageModel` class utilizing `nn.Embedding`, `nn.RNN` (`batch_first=True`), and `nn.Linear` layers.
* **Training & Evaluation:** A robust training loop computing Cross-Entropy Loss and perplexity. 
* **Text Generation:** Custom functions to sample the next token and generate sequences of 30+ tokens.

## Methodology & Key Findings

### Data Preparation
* **Leakage Prevention:** Data was split by **sentence** (80% Train, 10% Validation, 10% Test) rather than by token to ensure fragments of the same sequence did not artificially inflate test performance.

### Statistical Model Performance
* The Trigram model achieved a test perplexity of ~1590. 
* **Analysis:** It captured local grammar syntax well (e.g., generating highly accurate short noun phrases) but failed at global coherence, frequently shifting topics abruptly due to its inability to track long-range dependencies.

### Neural Model Performance (Overfitting)
* The PyTorch RNN achieved an excellent training loss (0.38) but a skyrocketing test perplexity (~15749). 
* **Analysis:** This proves that a highly parameterized RNN (~2.1 Million parameters) will severely overfit when trained on a small dataset (~4,600 sentences) without heavy regularization (like Dropout or weight decay). 
* **Text Generation:** When forced to generate 30+ tokens (ignoring `<eos>`), the RNN showed repetitive looping behaviors and completely failed to maintain a logical thought, highlighting the limitations of simple, unregularized RNNs for robust text generation.

- Completed with the help of google colab and Gemini
