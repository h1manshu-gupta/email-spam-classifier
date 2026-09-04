# 📧 Email/SMS Spam Classifier

A Machine Learning and Natural Language Processing (NLP) project that classifies text messages as **Spam** or **Ham (Not Spam)**.

The project compares multiple machine learning algorithms using two text feature extraction techniques:

- CountVectorizer
- TF-IDF Vectorizer

After evaluating all models using **Accuracy, Precision, Recall, and F1-Score**, **TF-IDF + Bernoulli Naive Bayes** was selected as the final model and deployed using **Streamlit**.

---

## 🚀 Application Preview

The Streamlit application allows users to enter an Email or SMS message and receive a prediction.

### Example

**Input:**

> Hey! I wanted to connect with you regarding our meeting.

**Prediction:**

> ✅ Not Spam

The application also displays the predicted probability of the message being classified as spam.

---

## 🚀 Live Demo

[Click here to try the Email/SMS Spam Classifier](https://emailmessage-spam-classifier.streamlit.app/)

--

# 📌 Project Workflow

```text
Raw Email/SMS Message
        │
        ▼
Text Preprocessing
        │
        ├── Convert to Lowercase
        ├── Tokenization
        ├── Remove Non-Alphanumeric Characters
        ├── Stopword Removal
        └── Stemming
        │
        ▼
Feature Extraction
        │
        ├── CountVectorizer
        └── TF-IDF Vectorizer
        │
        ▼
Machine Learning Models
        │
        ├── Decision Tree
        ├── Random Forest
        ├── Gaussian Naive Bayes
        ├── Bernoulli Naive Bayes
        └── Multinomial Naive Bayes
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        └── F1-Score
        │
        ▼
Best Model Selection
        │
        ▼
TF-IDF + Bernoulli Naive Bayes
        │
        ▼
Streamlit Deployment
```

---

# 📂 Project Structure

```text
spam_detection/
│
├── data/
│   ├── data_transformed.csv
│   ├── SpamData
│   └── SpamData.csv
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   ├── eda.ipynb
│   ├── models.ipynb
│   ├── preprocessing.ipynb
│   └── text_cleaning.ipynb
│
├── app.py
│
└── README.md
```

### Folder Description

| Folder/File | Description |
|---|---|
| `data/` | Contains the original and processed datasets |
| `models/` | Contains the trained and serialized machine learning model |
| `notebooks/` | Contains notebooks for EDA, preprocessing, text cleaning, and model training |
| `app.py` | Streamlit application for deploying the spam classifier |
| `README.md` | Project documentation |

---

# 🧹 Text Preprocessing

Before training the machine learning models, the text messages were cleaned and transformed.

The preprocessing steps include:

1. Convert text to lowercase.
2. Tokenize the text using NLTK.
3. Keep only alphanumeric tokens.
4. Remove English stopwords.
5. Apply stemming using `PorterStemmer`.

### Example

**Original Message:**

```text
Congratulations! You have won a free iPhone. Click here to claim now!
```

**Processed Message:**

```text
congratul free iphon click claim
```

---

# 🔤 Feature Extraction

Two feature extraction techniques were used to convert text into numerical features.

## CountVectorizer

CountVectorizer converts text into a numerical representation based on the frequency of words.

Example:

```text
Message: "free free offer"

Vocabulary:
free   offer

Vector:
[2, 1]
```

---

## TF-IDF Vectorizer

TF-IDF assigns importance to words based on:

- Term Frequency (TF)
- Inverse Document Frequency (IDF)

It reduces the importance of commonly occurring words and gives more importance to words that are informative for classification.

The TF-IDF vectorizer used:

```python
TfidfVectorizer(max_features=3000)
```

---

# 🤖 Machine Learning Models Evaluated

The following machine learning algorithms were evaluated:

| Algorithm | Description |
|---|---|
| Decision Tree (DT) | Tree-based classification algorithm |
| Random Forest (RF) | Ensemble of multiple decision trees |
| Gaussian Naive Bayes (GNB) | Naive Bayes classifier based on Gaussian distribution |
| Bernoulli Naive Bayes (BNB) | Naive Bayes classifier suitable for binary/discrete features |
| Multinomial Naive Bayes (MNB) | Naive Bayes classifier commonly used for text classification |

Each algorithm was evaluated using both:

- CountVectorizer
- TF-IDF Vectorizer

---

# 📊 Model Performance

## CountVectorizer Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.977756 | 1.000000 | 0.824427 | 0.903766 |
| BernoulliNB | 0.971954 | 1.000000 | 0.778626 | 0.875536 |
| MultinomialNB | **0.987427** | 0.960938 | **0.938931** | **0.949807** |
| Decision Tree | 0.930368 | 0.953846 | 0.473282 | 0.632653 |
| GaussianNB | 0.876209 | 0.506329 | 0.916031 | 0.652174 |

### 🏆 Best CountVectorizer Model

**CountVectorizer + Multinomial Naive Bayes**

| Metric | Score |
|---|---:|
| Accuracy | **98.74%** |
| Precision | **96.09%** |
| Recall | **93.89%** |
| F1-Score | **94.98%** |

Multinomial Naive Bayes achieved the best overall performance using CountVectorizer, with the highest accuracy, recall, and F1-score among the CountVectorizer models.

---

## TF-IDF Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.980658 | 1.000000 | 0.847328 | 0.917355 |
| MultinomialNB | 0.978723 | 1.000000 | 0.832061 | 0.908333 |
| BernoulliNB | **0.988395** | **0.991736** | **0.916031** | **0.952381** |
| Decision Tree | 0.939072 | 0.869565 | 0.610687 | 0.717489 |
| GaussianNB | 0.849130 | 0.451362 | 0.885496 | 0.597938 |

### 🏆 Best TF-IDF Model

**TF-IDF + Bernoulli Naive Bayes**

| Metric | Score |
|---|---:|
| Accuracy | **98.84%** |
| Precision | **99.17%** |
| Recall | **91.60%** |
| F1-Score | **95.24%** |

---

# 🏆 Final Model Selection

The final model selected for deployment is:

## TF-IDF + Bernoulli Naive Bayes

```text
TF-IDF Vectorizer
        +
Bernoulli Naive Bayes
```

### Final Performance

| Metric | Score |
|---|---:|
| Accuracy | **98.84%** |
| Precision | **99.17%** |
| Recall | **91.60%** |
| F1-Score | **95.24%** |

This combination achieved the highest overall **Accuracy and F1-Score** among all evaluated models.

Although **CountVectorizer + Multinomial Naive Bayes** achieved slightly higher recall, **TF-IDF + Bernoulli Naive Bayes** provided the best overall balance between precision and recall.

The final model was therefore selected based on its strong overall performance and high F1-score.

---

# 📈 Key Observations

## Naive Bayes Models Performed Best

BernoulliNB and MultinomialNB consistently achieved the strongest performance across both vectorization techniques.

This is expected because Naive Bayes algorithms are well suited for:

- High-dimensional data
- Sparse feature matrices
- Word frequency-based features
- Text classification problems

---

## Random Forest Achieved Perfect Precision

Random Forest achieved:

```text
Precision = 100%
```

for both CountVectorizer and TF-IDF.

However, its recall was lower than the best Naive Bayes models.

This means that Random Forest was highly conservative when predicting spam. When it predicted a message as spam, it was correct, but it missed more actual spam messages.

---

## GaussianNB Performed Poorly

GaussianNB achieved relatively high recall but significantly lower precision.

For example, with TF-IDF:

```text
Precision = 45.14%
Recall = 88.55%
```

This indicates that GaussianNB detected many spam messages but incorrectly classified many legitimate messages as spam.

GaussianNB is generally not an ideal choice for sparse text features because it assumes that features follow a Gaussian distribution.

---

## Decision Tree Had Lower Recall

Decision Tree performed better than GaussianNB in terms of precision but had significantly lower recall compared to the Naive Bayes models.

This indicates that the Decision Tree missed a larger number of actual spam messages.

---

# 🌐 Streamlit Application

The final model is deployed using Streamlit.

The application allows users to:

1. Enter an Email or SMS message.
2. Click the prediction button.
3. Receive a Spam or Not Spam classification.
4. View the predicted spam probability.

The application applies the same preprocessing steps used during model training before sending the processed message to the trained model.

### Application Flow

```text
User Input
    │
    ▼
Text Preprocessing
    │
    ▼
TF-IDF Vectorizer
    │
    ▼
Bernoulli Naive Bayes
    │
    ▼
Spam / Not Spam Prediction
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Joblib

---

# 📚 Machine Learning and NLP Concepts Used

This project covers the following concepts:

- Exploratory Data Analysis
- Text Cleaning
- Natural Language Processing
- Tokenization
- Stopword Removal
- Stemming
- Bag of Words
- CountVectorizer
- TF-IDF
- Naive Bayes Classification
- Decision Trees
- Random Forest
- Model Evaluation
- Accuracy
- Precision
- Recall
- F1-Score
- Precision-Recall Trade-off
- Machine Learning Pipelines
- Model Deployment

---

# 👨‍💻 Author

**Himanshu Gupta**

---

## ⭐ If You Found This Project Interesting

Consider giving the repository a star!
