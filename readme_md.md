# 📱 SMS Spam Detection

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies SMS messages as either **Spam** (unwanted/promotional) or **Ham** (legitimate). This project includes a user-friendly web interface built with Streamlit.

## 🚀 Project Overview

With the rise of digital communication, spam messages have become a significant nuisance and security risk. This project solves this real-world problem by taking an SMS message as input and predicting its classification using a trained Naive Bayes classifier.

### Key Features
*   **Text Preprocessing:** Cleans raw text data using standard NLP techniques.
*   **Feature Extraction:** Converts text into numerical data using TF-IDF (Term Frequency-Inverse Document Frequency).
*   **High Accuracy Model:** Utilizes a Naive Bayes classifier achieving **97% accuracy** and **100% precision**.
*   **Interactive Web App:** A Streamlit-based frontend for real-time message classification.

---

## 🛠️ Tech Stack & Libraries
*   **Language:** Python
*   **Frontend:** Streamlit (`st`)
*   **Machine Learning:** Scikit-Learn (Naive Bayes, TF-IDF Vectorizer)
*   **NLP Processing:** NLTK (Stopwords, PorterStemmer)
*   **Model Deployment:** Pickle (for saving/loading `model.pkl` and `vectorizer.pkl`)

---

## 📊 Dataset & Preprocessing

The model was trained on a dataset of over 5,500 SMS messages. To ensure high data quality, the following steps were taken:
1.  **Data Cleaning:** Removed irrelevant columns, dropped duplicate messages, and encoded labels numerically (Spam = 1, Ham = 0). The final dataset resulted in 5,169 unique messages.
2.  **Exploratory Data Analysis (EDA):** Analyzed class distributions, word counts, and visual patterns using word clouds and histograms.
3.  **NLP Pipeline:**
    *   **Lowercasing:** Standardized all text to lowercase.
    *   **Tokenization:** Split sentences into individual words.
    *   **Stopword Removal:** Filtered out common English words (e.g., "a", "the", "is") using NLTK.
    *   **Punctuation Removal:** Stripped out special characters to reduce noise.
    *   **Stemming:** Used `PorterStemmer` to reduce words to their root form (e.g., "playing" -> "play").
4.  **Vectorization:** Transformed the cleaned text tokens into a sparse numerical matrix (~6,700 features) using a `TfidfVectorizer`.

---

## 🧠 Model Performance

The project evaluated multiple classifiers, ultimately selecting a **Naive Bayes** model due to its high performance on sparse TF-IDF text features. 

*   **Accuracy:** 97%
*   **Precision:** 100% (Zero false positives—meaning no legitimate messages are accidentally classified as spam).

---

## 💻 How to Run the Project Locally

### Prerequisites
Make sure you have Python installed on your system. 

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install streamlit scikit-learn nltk
   ```

4. **Download NLTK Data (Run this inside a Python shell once):**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## 📸 Screenshots

### Web Application Interface
*(You can upload your `Screenshot 2026-11-18 230456.png` to your GitHub repo and link it here)*

![App Screenshot](./Screenshot 2026-11-18 230456.png)

*The app displays the transformed text tokens, the TF-IDF feature shape, prediction probabilities, and the final visual alert for Spam or Ham.*

---

## 📂 Repository Structure

```text
├── app.py                 # Main Streamlit application script
├── model_1.pkl            # Trained Machine Learning Model (Pickle file)
├── vectorizer_1.pkl       # Fitted TF-IDF Vectorizer (Pickle file)
├── README.md              # Project Documentation
└── (Optional Jupyter Notebooks used for EDA and Model Training)
```