# AIG230 Assignment 6: Natural Language Processing
**By:** Aziz Rahman  

## Project Overview
This repository contains the implementation and analysis of two language models: a Statistical Trigram Language Model and a Recurrent Neural Network (RNN) Language Model. Both models were trained to predict the next token in a sequence using the NLTK Brown corpus (news category). 

The purpose of this project is to compare the performance, text generation capabilities, and perplexity of statistical vs. neural approaches on a relatively small dataset.

---

## Part D: Data Preparation & Vocabulary
To ensure a fair comparison, both models share the exact same preprocessing pipeline, vocabulary, and data splits.

* **Dataset:** NLTK Brown corpus (`categories='news'`).
* **Preprocessing:** Tokens were lowercased, punctuation-only tokens were removed, and stopwords were retained. Special tokens (`<bos>`, `<eos>`, `<unk>`) were added to robustly handle boundaries and unknown words.
* **Data Split:** The data was strictly split by **sentence** (not by token) to prevent data leakage across splits. The split ratio used was 80% Train, 10% Validation, and 10% Test.
* **Vocabulary:** Built exclusively from the training split with a minimum frequency (`min_freq`) of 2, resulting in a vocabulary size of 5,353 tokens.

**Key Finding:** Splitting by sentence rather than by token is a critical step to ensure that fragments of the same sequence do not artificially inflate test performance.

---

## Part A: Statistical Trigram Language Model
A baseline trigram model was trained using the `nltk.lm` library.

* **Smoothing:** Laplace (Add-1) smoothing was applied to assign a small, non-zero probability to unseen n-grams, preventing infinite perplexity when encountering new word sequences.
* **Perplexity:** * Validation Perplexity: 1647.92
  * Test Perplexity: 1590.38
* **Text Generation Analysis:** The generated text samples are locally grammatical (e.g., "the potato chip industry"), demonstrating that trigrams successfully capture short syntax. However, the model lacks global coherence and long-range dependency, frequently shifting topics abruptly mid-sentence.

---

## Part B: Neural Language Model (RNN)
A custom Recurrent Neural Network was implemented using PyTorch, trained using a next-token prediction format.

* **Architecture:** * `nn.Embedding` (Dimension: 128)
  * `nn.RNN` (Hidden Dimension: 256, `batch_first=True`)
  * `nn.Linear` (Output Dimension: 5,353)
* **Total Parameters:** ~2.15 Million
* **Perplexity:**
  * Final Test Perplexity: 5939.77
* **Training Analysis (Overfitting):** While the training loss steadily dropped to 0.51, the validation and test perplexities exploded. This indicates severe overfitting. Because the model is highly parameterized but trained on a very small dataset (~4,600 sentences), it memorized the training sequences rather than learning generalizable patterns. 
* **Text Generation Analysis:** The RNN text generation showed repetitive looping behaviors (e.g., repeating "and mrs. [name]" multiple times). It struggled with long-range dependencies, failing to resolve the subject of the sentence and relying heavily on local, repetitive patterns.

---

## Conclusion
For small datasets like the Brown news category, a smoothed Statistical N-gram model significantly outperforms a simple, unregularized RNN in terms of test perplexity. The RNN requires either a much larger dataset or strong regularization techniques (like Dropout or weight decay) to prevent overfitting and improve generalization.

- Completed with the help of google colab and Gemini
