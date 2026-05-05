from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__, static_folder='.')

df = pd.read_csv('dataset.tsv', sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(clean_text)

vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_message']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
MODEL_ACCURACY = round(accuracy_score(y_test, y_pred) * 100, 2)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message.strip():
        return jsonify({'error': 'Empty message'}), 400
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = round(max(proba) * 100, 1)
    return jsonify({
        'result': 'spam' if result == 1 else 'ham',
        'confidence': confidence,
        'model_accuracy': MODEL_ACCURACY
    })

if __name__ == '__main__':
    print(f"Model trained! Accuracy: {MODEL_ACCURACY}%")
    app.run(debug=True, port=5000)