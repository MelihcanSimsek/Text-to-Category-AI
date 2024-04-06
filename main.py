from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from vnlp import Normalizer
from vnlp import StopwordRemover

stopword_remover = StopwordRemover()
normalizer = Normalizer()
app_port =5000
app = Flask(__name__)
CORS(app,resources={r"/predict_category": {"origins": "http://localhost:4200"}})

model = joblib.load('ensemble_model.joblib')

df = pd.read_excel("updated_dataset.xlsx")
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df["text"])

def clean_text(text):
    lower_text = normalizer.lower_case(text)
    text_without_accent_marks = normalizer.remove_accent_marks(lower_text)
    text_without_punctuations = normalizer.remove_punctuations(text_without_accent_marks)
    text_without_numbers = normalizer.convert_numbers_to_words(text_without_punctuations.split(), 5, ".")
    text_deasciified = normalizer.deasciify(text_without_numbers)
    text_corrected = normalizer.correct_typos(text_deasciified)
    text_without_stopwords = stopword_remover.drop_stop_words(text_corrected)
    return " ".join(text_without_stopwords)

@app.route('/predict_category', methods=['POST'])
def predict_category():
    try:
        data = request.get_json()
        cleaned_text = clean_text(data['text'])
        text_vector = tfidf.transform([cleaned_text])
        prediction = model.predict(text_vector)[0]
        return jsonify({'category': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True,port=app_port)
