# SMS Spam/Ham Classifier

This project implements a highly accurate text classification model to distinguish between legitimate (Ham) and unsolicited commercial (Spam) text messages. The classifier is built using Python, scikit-learn, and fundamental Natural Language Processing (NLP) techniques.

## Project Goal

To build and evaluate a robust machine learning model capable of achieving high precision and recall in filtering SMS spam, minimizing **False Positives** (flagging legitimate messages as spam).

## Performance Summary

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Overall Accuracy** | ~98.5% | Achieved on the test set. |
| **Classification Model** | Multinomial Naive Bayes (MNB) | Ideal for sparse text data. |
| **Feature Extraction** | TF-IDF Vectorization | Used to weight word importance. |
| **False Positives (FP)** | Very Low | Essential for a functional spam filter. |

---

## Technical Stack

* **Language:** Python 3.x
* **Libraries:**
    * `pandas` (Data handling)
    * `scikit-learn` (Model training and evaluation)
    * `nltk` (Natural Language Toolkit for preprocessing)
    * `pickle` (Model serialization)

---

## Getting Started

### Prerequisites

You need Python 3.x and the required libraries installed.

```bash
pip install pandas scikit-learn nltk
