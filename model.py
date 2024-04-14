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

# import joblib
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import VotingClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.svm import SVC
# from vnlp import Normalizer
# from vnlp import StopwordRemover

# stopword_remover = StopwordRemover()
# normalizer = Normalizer()

# df = pd.read_excel("updated_dataset.xlsx")

# X = df["text"]
# y = df["category"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# tfidf = TfidfVectorizer(max_features=1000)
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)

# parameters = {'alpha': [0.1, 0.5, 1.0, 2.0]}
# nb_model = GridSearchCV(MultinomialNB(), parameters, cv=5)
# nb_model.fit(X_train_tfidf, y_train)

# models = [('nb', nb_model), ('lr', LogisticRegression()), ('svm', SVC())]
# ensemble = VotingClassifier(models)
# ensemble.fit(X_train_tfidf, y_train)

# y_pred = ensemble.predict(X_test_tfidf)
# accuracy = accuracy_score(y_test, y_pred)
# kategori_isimleri = df["category"].unique()
# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm, index=kategori_isimleri, columns=kategori_isimleri)

# print("Confusion Matrix:")
# print(cm_df)
# print(f"Doğruluk Oranı: {accuracy:.2%}")
# joblib.dump(ensemble, 'ensemble_model.joblib')

