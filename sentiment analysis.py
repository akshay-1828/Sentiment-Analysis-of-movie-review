from nltk.corpus import movie_reviews, stopwords
from nltk import download
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Downloads
download('movie_reviews')
download('stopwords')

# Prepare data
docs = []
labels = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        docs.append(movie_reviews.raw(fileid))
        labels.append(category)

# Convert labels to binary
y = [1 if label == 'pos' else 0 for label in labels]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(docs)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# ---- SENTIMENT ANALYSIS ----
# Short review sets
short_positive = {"good", "great", "amazing", "awesome", "loved", "nice"}
short_negative = {"bad", "boring", "worst", "terrible", "awful", "hate"}

# Get custom review
custom_review = input("\nEnter your own review to analyze its sentiment:\nYour review: ")

# Handle short reviews
if len(custom_review.split()) < 4:
    sentiment = "pos" if any(word in custom_review.lower() for word in short_positive) \
                     else "neg" if any(word in custom_review.lower() for word in short_negative) \
                     else "unknown"
    print("Predicted (short review logic):", sentiment)
else:
    custom_vector = vectorizer.transform([custom_review])
    custom_pred = model.predict(custom_vector)
    print("Predicted Sentiment:", "pos" if custom_pred[0] == 1 else "neg")
