import joblib
import pandas as pd
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

df = pd.read_excel("updated_dataset.xlsx")

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df["text"])

parameters = {'alpha': [0.1, 0.5, 1.0, 2.0]}
nb_model = GridSearchCV(MultinomialNB(), parameters, cv=5)
nb_model.fit(X_tfidf, df["category"])

models = [('nb', nb_model), ('lr', LogisticRegression()), ('svm', SVC())]
ensemble = VotingClassifier(models)
ensemble.fit(X_tfidf, df["category"])

y_true = df["category"]

y_pred = ensemble.predict(X_tfidf)
accuracy = accuracy_score(y_true, y_pred)
kategori_isimleri = df["category"].unique()
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=kategori_isimleri, columns=kategori_isimleri)

print(cm_df)
print(f"Doğruluk Oranı: {accuracy:.2%}")
joblib.dump(ensemble, 'ensemble_model.joblib')

