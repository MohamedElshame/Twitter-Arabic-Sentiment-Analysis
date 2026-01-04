# Arabic Sentiment Analysis Project

## Project Documentation

---

## 1. Executive Summary

This project implements a **sentiment analysis system for Arabic text** using both **Traditional Machine Learning** and **Deep Learning** approaches. The system classifies Arabic text into four sentiment categories:

* **Positive**
* **Negative**
* **Objective**
* **Neutral**

The final deployed model achieves an accuracy of **88.04%** using the **AraBERT** transformer model on **3-class classification**, demonstrating strong performance in understanding Arabic sentiment.

---

## 2. Project Overview

### 2.1 Objective

The main goal of this project is to develop an accurate and reliable Arabic sentiment classification system that can:

* Classify Arabic text into sentiment categories
* Handle Arabic language complexities and dialects
* Provide confidence scores for predictions
* Compare multiple modeling approaches

### 2.2 Scope

The project covers the complete NLP pipeline, including:

* Arabic text preprocessing
* Traditional Machine Learning models
* Deep Learning using transformer models
* Model evaluation and comparison
* Production-ready inference pipeline

### 2.3 Target Use Cases

* Social media monitoring
* Customer feedback analysis
* Brand sentiment tracking
* Opinion mining from Arabic content

---

## 3. Dataset Description

### 3.1 Source

The dataset is a **hybrid collection** consisting of:

* Approximately **20,000 tweets** collected from Twitter
* Additional labeled Arabic text samples from public datasets

All samples are manually or automatically labeled for **supervised learning**.

### 3.2 Dataset Statistics

| Metric                     | Value              |
| -------------------------- | ------------------ |
| Original samples           | 41,644             |
| Samples after cleaning     | 41,486             |
| Balanced samples (3-class) | 32,940             |
| Balanced samples (4-class) | 11,704             |
| Text length range          | 3 – 400 characters |

### 3.3 Label Distribution

#### Four-Class Classification

* **Positive (POS):** 38.98%
* **Negative (NEG):** 34.62%
* **Objective (OBJ):** 19.37%
* **Neutral (NEUTRAL):** 7.03%

#### Three-Class Classification (Final Model)

* **Positive (POS)**
* **Negative (NEG)**
* **Objective (OBJ)** *(Neutral merged into Objective)*

### 3.4 Data Split

| Dataset    | Percentage | Purpose                                |
| ---------- | ---------- | -------------------------------------- |
| Training   | 90%        | Model training                         |
| Validation | 5%         | Hyperparameter tuning & early stopping |
| Test       | 5%         | Final evaluation                       |

---

## 4. Methodology

### 4.1 Data Preprocessing

The preprocessing pipeline includes the following steps:

#### 1. Text Normalization

* Removal of diacritics (Tashkeel)
* Removal of elongation (Tatweel)
* Normalization of Arabic characters (Hamza, Lam-Alef)
* Basic spelling normalization

#### 2. Token Handling

* User mentions replaced with placeholder tokens
* Numbers replaced with placeholder tokens
* URLs removed
* English characters removed
* Special characters and emojis cleaned

#### 3. Data Balancing

* Undersampling applied to majority classes
* Equal class distribution achieved

### 4.2 Feature Engineering

#### Machine Learning Models

* TF-IDF vectorization
* Unigrams and bigrams
* Maximum document frequency threshold: **60%**

#### Deep Learning Model

* AraBERT tokenizer
* Maximum sequence length: **256 tokens**
* Padding and truncation applied

### 4.3 Models Implemented

#### Traditional Machine Learning

1. Linear Support Vector Classifier (LinearSVC)
2. Logistic Regression
3. Random Forest Classifier
4. Multinomial Naive Bayes

#### Deep Learning

* **AraBERT** (`aubmindlab/bert-base-arabertv02`)
* Pre-trained transformer model specialized for Arabic

### 4.4 Training Configuration

#### Machine Learning

* Stratified data splitting
* Default hyperparameters

#### AraBERT

* Learning rate: `2e-5`
* Batch size: `16`
* Epochs: `3`
* Weight decay: `0.01`
* Early stopping enabled

---

## 5. Results

### 5.1 Model Performance Comparison (3-Class)

| Rank | Model               | Accuracy   |
| ---- | ------------------- | ---------- |
| 1    | AraBERT             | **88.04%** |
| 2    | LinearSVC           | 84.76%     |
| 3    | Logistic Regression | 81.42%     |
| 4    | Random Forest       | 81.12%     |
| 5    | Multinomial NB      | 80.15%     |
| 6    | Baseline (Majority) | 33.33%     |

### 5.2 AraBERT Detailed Metrics

| Metric              | Value      |
| ------------------- | ---------- |
| **Accuracy**        | 88.04%     |
| **Precision**       | 0.8810     |
| **Recall**          | 0.8804     |
| **F1-Score**        | 0.8805     |

### 5.3 Key Findings

1. AraBERT outperforms traditional ML models by **3.28%** over LinearSVC
2. Three-class classification performs better than four-class
3. LinearSVC is the strongest traditional ML model
4. All models significantly outperform the baseline

### 5.4 Improvement Over Baseline

The final model achieves a **+54.71% accuracy improvement** over random classification.

---

## 6. Key Features

### 6.1 Arabic Language Support

* Arabic-specific preprocessing
* Support for MSA and dialectal Arabic
* Character normalization and noise handling

### 6.2 Dual Modeling Strategy

* Traditional ML for speed and interpretability
* Deep Learning for maximum accuracy
* Comparative evaluation

### 6.3 Comprehensive Evaluation

* Accuracy, Precision, Recall, F1-score
* Confusion matrix analysis
* Error analysis with examples

### 6.4 Production-Ready Prediction

* Confidence score per prediction
* Class probability distribution
* Confidence level indicator

### 6.5 Model Persistence

* Saved trained models
* Saved tokenizer
* Ready for deployment

---

## 7. Project Structure

### 7.1 Notebook Organization

| Section              | Description             |
| -------------------- | ----------------------- |
| Data Loading         | Import and exploration  |
| Exploratory Analysis | Data statistics         |
| Preprocessing        | Text cleaning           |
| Data Balancing       | Class balancing         |
| ML Training          | Traditional ML models   |
| AraBERT Training     | Transformer fine-tuning |
| Evaluation           | Metrics & analysis      |
| Prediction           | Inference pipeline      |

### 7.2 Output Files

| File                             | Description           |
| -------------------------------- | --------------------- |
| `cleaned_dataset.csv`            | Preprocessed dataset  |
| `arabert_finetuned_multi`        | 4-class AraBERT model |
| `arabert_finetuned_3class_final` | Best 3-class model    |

---

## 8. Requirements

### 8.1 Dependencies

* pandas
* numpy
* matplotlib
* scikit-learn
* transformers
* torch
* datasets
* aranorm
* emoji
* tqdm

### 8.2 Hardware Requirements

* GPU recommended for AraBERT training
* Minimum **8GB RAM** for ML models
* **16GB RAM** recommended for transformer training

---

## 9. Usage

### 9.1 Running the Notebook

Execute cells sequentially to:

1. Load and preprocess data
2. Train models
3. Evaluate performance
4. Run prediction functions

### 9.2 Using Saved Models

* Load tokenizer from saved directory
* Load trained model
* Perform inference without retraining

### 9.3 Prediction Output

The prediction function returns:

* Predicted sentiment label
* Confidence score (0–1)
* Probability for each class
* Confidence level indicator

---

## 10. Conclusion

This project demonstrates a complete **Arabic Sentiment Analysis pipeline** using both classical and transformer-based approaches. The AraBERT model achieves **state-of-the-art performance** with **88.04% accuracy**, making it suitable for real-world deployment in Arabic NLP applications.

---

## 11. References

* AraBERT: *Pre-training BERT for Arabic Language Understanding*
* Scikit-learn Documentation
* Hugging Face Transformers Library
* aranorm: Arabic Text Normalization Library

---

**Author:** Mohamed Elshamy
**Version:** 1.0
**Date:** December 18, 2025
