# 📰 Fake News Detection using Machine Learning

An end-to-end **Fake News Detection** system built using **Natural Language Processing (NLP)** and **Machine Learning**. The application classifies news articles as **Fake** or **Real** using TF-IDF vectorization and multiple machine learning algorithms. The final model is deployed as an interactive web application using **Streamlit**.

---

## 🚀 Live Demo

🔗 **Live App:** *Paste your Streamlit URL here*

## 📂 GitHub Repository

🔗 *Paste your GitHub repository URL here*

---

## 📌 Project Overview

Fake news has become a major challenge in today's digital world. This project aims to automatically classify news articles as **Fake** or **Real** by applying text preprocessing, feature extraction, and machine learning techniques.

The project follows a complete machine learning workflow:

- Data Collection
- Exploratory Data Analysis (EDA)
- Text Preprocessing
- TF-IDF Feature Extraction
- Model Training & Evaluation
- Model Deployment using Streamlit

---

## 📊 Dataset

**Dataset:** Fake and Real News Dataset (Kaggle)

The dataset consists of two CSV files:

- **Fake.csv** – Fake news articles
- **True.csv** – Real news articles

### Dataset Statistics

| Description | Value |
|------------|------:|
| Total Articles | 44,898 |
| Fake News | 23,481 |
| Real News | 21,417 |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Joblib
- Streamlit

---

## 🔄 Workflow

```
News Article
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
Lemmatization
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
      │
      ▼
Prediction (Fake / Real)
```

---

## 🧹 Text Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove punctuation
- Remove numbers
- Remove extra spaces
- Tokenization
- Stopword removal
- Lemmatization

---

## 🤖 Machine Learning Models Compared

The following models were trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)
- Random Forest Classifier
- Passive Aggressive Classifier

---

## 📈 Model Performance

| Model | Accuracy |
|-------------------------|---------:|
| Random Forest | **99.78%** |
| Linear SVM | **99.48%** |
| Passive Aggressive | **99.43%** |
| Logistic Regression | **98.76%** |
| Multinomial Naive Bayes | **94.30%** |

---

## 📸 Application Screenshot

> *(Add a screenshot of your Streamlit application here)*

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── app.py
├── Fake_News_Detection.ipynb
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Fake-News-Detection.git
```

Move into the project directory:

```bash
cd Fake-News-Detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Sample Prediction

### Input

```
Scientists discover a magical fruit that allows humans to breathe underwater.
```

### Output

```
❌ Fake News
```

---

### Input

```
The Reserve Bank of India announced its latest monetary policy after the MPC meeting.
```

### Output

```
✅ Real News
```

---

## 📚 Key Learnings

Through this project, I learned:

- NLP preprocessing techniques
- TF-IDF feature extraction
- Training and comparing multiple machine learning models
- Model evaluation using Accuracy, Precision, Recall, and F1-score
- Saving and loading models using Joblib
- Building and deploying ML applications with Streamlit

---

## 🔮 Future Improvements

- Deep Learning-based models (LSTM/BERT)
- Transformer-based text classification
- Probability confidence visualization
- Docker deployment
- Cloud deployment using AWS or Azure

---

## 👤 Author

**Tanuja**

GitHub: https://github.com/YOUR_USERNAME

LinkedIn: *Add your LinkedIn profile here*
