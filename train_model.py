import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from joblib import dump

# Step 1: Dataset
data = {
    'review': [
        "This movie was fantastic! I loved it.",
        "Absolutely terrible. Waste of time.",
        "The plot was interesting and well-executed.",
        "Horrible acting and bad direction.",
        "Great cinematography and amazing soundtrack.",
        "Not my type of movie, too slow and boring.",
        "I enjoyed every moment of it!",
        "Disappointing ending but overall okay.",
        "Worst storyline ever, not recommended.",
        "Brilliant movie! The cast did an amazing job.",
        "Mediocre film, could have been better.",
        "Totally worth watching, would recommend.",
        "Awful movie with terrible editing.",
        "Loved the visuals and the background score.",
        "The acting was weak and the plot was dull."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)
df['sentiment'] = df['sentiment'].astype(int)

# Step 2: Split data into train/test manually
train_df = df.iloc[:12]
test_df = df.iloc[12:]

# Step 3: Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Step 4: Train model
pipeline.fit(train_df['review'], train_df['sentiment'])

# Step 5: Evaluate accuracy
preds = pipeline.predict(test_df['review'])
acc = accuracy_score(test_df['sentiment'], preds)
print(f"✅ Model Accuracy: {acc*100:.2f}%")

# Step 6: Save model
dump(pipeline, 'sentiment_model.joblib')
print("✅ Model trained successfully on full dataset and saved as 'sentiment_model.joblib'!")
