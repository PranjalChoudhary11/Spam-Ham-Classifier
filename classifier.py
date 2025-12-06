import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

REQUIRED_NLTK_RESOURCES = ['stopwords', 'punkt', 'punkt_tab'] 

for resource in REQUIRED_NLTK_RESOURCES:
    try:
        # Check if the resource is already downloaded
        nltk.data.find(f'tokenizers/{resource}') 
        nltk.data.find(f'corpora/{resource}') 
    except LookupError:
        print(f"Downloading NLTK resource: {resource}...")
        nltk.download(resource)
    except Exception as e:
        print(f"An unexpected error occurred during NLTK download for {resource}: {e}")
        
print("NLTK resources checked and downloaded if necessary.")

#  Data Loading  
try:
    # Uses Tab Separator and explicitly names the two columns we need.
    df = pd.read_csv(
        'spam.csv', 
        sep='\t', 
        encoding='latin-1', 
        header=None, 
        names=['label', 'text'] 
    ) 
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please provide the dataset in the same directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    print("CRITICAL: Please verify your 'spam.csv' file is the TSV format.")
    exit()

# Dropping duplicates
df.drop_duplicates(inplace=True)

# Convert labels to numerical format: 'ham' = 0, 'spam' =1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print("Initial data loading and cleaning complete.")
print(df.head())

# Text Preprocessing Function 

# Initialize Stemmer and Stopwords
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # check for non-string values
    if pd.isna(text):
        return ""
        
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    
    # Remove stopwords and apply stemming 
    processed_words = [
        ps.stem(word) for word in words 
        if word not in stop_words and word.isalnum()
    ]
    return ' '.join(processed_words)

# Apply the preprocessing function
df['clean_text'] = df['text'].apply(preprocess_text)

print("\nText preprocessing complete.")
print(df[['text', 'clean_text']].head())

# Feature Extraction and Split

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Model Training and Evaluation 

mnb_model = MultinomialNB()
mnb_model.fit(X_train, y_train)

y_pred = mnb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Evaluation")
print(f"Overall Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix")
print(conf_matrix)

tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nTrue Positives (Correctly identified SPAM): {tp}")
print(f"False Positives (HAM incorrectly labeled as SPAM): {fp}")

print("\n### Detailed Classification Report ###")
print(report)

# Prediction Function and Saving 

import pickle

def predict_email(email, vectorizer, model):
    clean_email = preprocess_text(email)
    vector = vectorizer.transform([clean_email]).toarray() 
    prediction = model.predict(vector)[0]
    return 'Spam' if prediction == 1 else 'Ham'

new_email_1 = "Win a free iPhone now! Click this link to claim your prize."
new_email_2 = "Meeting at 3 PM tomorrow regarding the project proposal."

print("\nExample Predictions")
print(f"'{new_email_1}' is: {predict_email(new_email_1, tfidf, mnb_model)}")
print(f"'{new_email_2}' is: {predict_email(new_email_2, tfidf, mnb_model)}")

pickle.dump(mnb_model, open('mnb_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

print("\nModel and Vectorizer saved as 'mnb_model.pkl' and 'tfidf_vectorizer.pkl'")