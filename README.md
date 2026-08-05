# 📰 Fake News Detection using Machine Learning

## About the Project

Fake news spreads quickly on social media and online platforms, making it difficult to verify whether the information is trustworthy. The goal of this project is to build a machine learning model that can classify a news article as **Fake** or **Real** using Natural Language Processing (NLP).

Instead of training just one model, I experimented with multiple machine learning algorithms, compared their performance, and selected the best-performing approach. I also built a simple Streamlit web application so users can paste any news article and instantly get a prediction.

---

## Live Demo

🔗 **Streamlit App:** [*Add your deployed app link here*
](https://epdgv3uffzn3thbn6zgao6.streamlit.app/)
---

## Dataset

This project uses the **Fake and Real News Dataset** available on Kaggle.

The dataset contains:

- **23,481** fake news articles
- **21,417** real news articles

for a total of **44,898** news articles.

---

## What I Did

The project follows a complete machine learning workflow:

- Explored and understood the dataset
- Cleaned and preprocessed the text
- Performed exploratory data analysis (EDA)
- Converted text into numerical features using TF-IDF
- Trained multiple machine learning models
- Compared model performance using different evaluation metrics
- Saved the best model
- Built and deployed a Streamlit application

---

## Text Preprocessing

Before training the model, each article was cleaned using the following steps:

- Converted text to lowercase
- Removed URLs and HTML tags
- Removed punctuation and numbers
- Removed stopwords
- Applied lemmatization
- Converted the cleaned text into TF-IDF vectors

---

## Models Compared

I trained and evaluated the following models:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)
- Random Forest Classifier
- Passive Aggressive Classifier

---

## Results

| Model | Accuracy |
|-------------------------|---------:|
| Random Forest | **99.78%** |
| Linear SVM | **99.48%** |
| Passive Aggressive | **99.43%** |
| Logistic Regression | **98.76%** |
| Multinomial Naive Bayes | **94.30%** |

Random Forest achieved the highest accuracy on this dataset. During the project, I also compared the strengths of different algorithms instead of relying on a single model.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Streamlit
- Joblib

---

## Project Structure

```text
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

## Sample Prediction

**Input**

```
Scientists discover a magical fruit that allows humans to breathe underwater.
```

**Prediction**

```
❌ Fake News
```

---

## What I Learned

This project helped me understand:

- The complete NLP pipeline using traditional machine learning
- Text preprocessing techniques
- TF-IDF feature extraction
- Comparing multiple machine learning models
- Evaluating models using Accuracy, Precision, Recall and F1-score
- Deploying a machine learning model using Streamlit

---

## Future Improvements

Some ideas to improve this project in the future:

- Fine-tune hyperparameters
- Experiment with deep learning models like LSTM
- Build a BERT-based classifier
- Add confidence scores and better visualizations
- Deploy using Docker and cloud platforms

---
