print("importing libraries")
from pickle import dump as pickle_dump
from pandas import read_csv as pd_read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# get data from csv
print("getting data from csv")
data = pd_read_csv("dataset.csv")
X_train, X_test, Y_train, Y_test = train_test_split(
    data["text"],
    data["text_type"],
    test_size=0.3,
)

# extract features
print("extracting features")
vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
X_train_vectorized.toarray().shape

# create and train model
print("training model")
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)
predictions = model.predict(vectorizer.transform(X_test))
print("Accuracy:", 100 * sum(predictions == Y_test) / len(predictions), "%")

# save model and vectorizer
print("saving model and vectorizer")
with open("models/model.pkl", "wb") as f:
    pickle_dump(model, f)
with open("models/vectorizer.pkl", "wb") as f:
    pickle_dump(vectorizer, f)
print("done")
