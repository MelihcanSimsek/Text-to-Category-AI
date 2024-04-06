from vnlp import Normalizer
from vnlp import StopwordRemover
import pandas as pd
import concurrent.futures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import re

df = pd.read_excel("dataset.xlsx")
stopword_remover = StopwordRemover()
normalizer = Normalizer()

def clean_text(text):
    lowerText = normalizer.lower_case(text)
    textListWithoutAccentMarkt = normalizer.remove_accent_marks(lowerText)
    textListWithoutPunctutations =  normalizer.remove_punctuations(textListWithoutAccentMarkt)
    textListwithConvertedNumber =  normalizer.convert_numbers_to_words(textListWithoutPunctutations.split(),5,".")
    textListWithDeasciification =  normalizer.deasciify(textListwithConvertedNumber)
    textListWithCorrectWords =  normalizer.correct_typos(textListWithDeasciification)
    textListWithoutStopWords =  stopword_remover.drop_stop_words(textListWithCorrectWords)
    return  " ".join(textListWithoutStopWords) 

for index, row in df.iterrows():
    df.at[index, "text"] = clean_text(row["text"])
    print(f"Processed row at index {index+1}")

df.to_excel("updated_dataset.xlsx", index=False)