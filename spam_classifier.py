import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('punkt_tab')

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
print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_spam(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned]).toarray()
    result = model.predict(vectorized)
    return "SPAM" if result[0] == 1 else "NOT SPAM"

print(predict_spam("Congratulations! You won a FREE iPhone. Click now!"))
print(predict_spam("Hey, are we still meeting tomorrow for lunch?"))
print(predict_spam("URGENT! Your bank account is blocked. Call now!"))