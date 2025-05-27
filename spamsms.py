import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
# Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
# Keep only required columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
# Encode labels: 'ham' → 0, 'spam' → 1
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # ham = 0, spam = 1
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)
# Convert text messages to TF-IDF feature vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# Predict on test data
y_pred = model.predict(X_test_tfidf)
# Evaluate the model
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
