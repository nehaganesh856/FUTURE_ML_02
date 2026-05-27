import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
df = pd.read_csv('tickets.csv') 
print(df.columns)
df['Ticket ID']=df['Ticket Type'].astype(str).fillna('')
print("columns:",df.columns.tolist())
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Ticket ID'])
y_category = df['Ticket ID']
X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2)
model = MultinomialNB().fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))