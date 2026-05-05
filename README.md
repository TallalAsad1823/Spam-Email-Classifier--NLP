# Spam Email & SMS Classifier | NLP & Machine Learning

A modern, accurate, and user-friendly **Spam Detection System** built using Natural Language Processing (NLP) and Machine Learning. The model can classify messages as **Spam** or **Ham** (Not Spam) with high accuracy.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-FF9F00?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-000000?style=for-the-badge&logo=python&logoColor=white)

---

## 🌟 Features

- **High Accuracy Model** (~97-98% on test data)
- **Clean Text Preprocessing** (Tokenization, Lemmatization, Stopwords Removal)
- **Beautiful Modern Web Interface** with real-time prediction
- **Confidence Score** for every prediction
- **Responsive Design** (Mobile + Desktop friendly)
- **Command Line Interface** support
- **Fast & Lightweight** Flask backend

---

## 📊 Dataset

- **SMS Spam Collection Dataset** (5,572 messages)
- Publicly available dataset containing real SMS messages labeled as **spam** or **ham**.
- Used for training and evaluating the classifier.

---

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Web Framework**: Flask
- **Machine Learning**: scikit-learn (Multinomial Naive Bayes)
- **NLP Library**: NLTK
- **Frontend**: HTML5, CSS3, Tailwind CSS (Custom)
- **Vectorization**: CountVectorizer (Bag of Words)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/TallalAsad1823/Spam-Email-Classifier--NLP.git
cd Spam-Email-Classifier--NLP

---

### 2. Install Dependencies
Bashpip install -r requirements.txt

---

### 3. Run the Application
Web Interface:
python app.py

---

### Open your browser and go to: http://127.0.0.1:5000
Command Line Mode:
python spam_classifier.py

📁 Project Structure
textSpam-Email-Classifier--NLP/
├── app.py                    # Flask Web Application
├── spam_classifier.py        # Model Training & CLI
├── index.html                # Frontend UI
├── dataset.tsv               # SMS Spam Dataset
├── requirements.txt          # Python dependencies
└── README.md

📈 Model Performance

Algorithm: Multinomial Naive Bayes
Accuracy: 97.5%+ (on test set)
Vectorization: Bag of Words (3000 features)
Preprocessing: Lowercasing, Lemmatization, Stopwords Removal




👨‍💻 Author
Tallal Asad

GitHub: @TallalAsad1823


📄 License
This project is open-source and available under the MIT License.
