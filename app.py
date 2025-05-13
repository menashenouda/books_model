import re
import random
from flask import Flask, request, jsonify
from transformers import pipeline
from spellchecker import SpellChecker
from rapidfuzz import process, fuzz
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Copy all your existing constants and functions here
INTENTS = {
    "greeting": {
        "patterns": [r"hi|hello|hey|good morning|good evening|how are you|how are you doing"],
        "responses": ["Hello! How can I help you today?", "Hi there! Ready for books recommendations?","Hey, hope you are doing well", "Hey, are you ready to read new books today"]
    },
    "leaving": {
        "patterns": [r"bye|goodbye|exit|quit|see you"],
        "responses": ["Goodbye! Happy reading!", "See you later!"]
    },
    "services": {
        "patterns": [r"what can you do|service|help"],
        "responses": ["I can recommend books! Just ask for recommendations and mention a book you like.","I am here to help you finding new books"]
    },
    "recommendation": {
        "patterns": [r"recommend|suggest|books like|a book like"],
    },
    "non_related": {
        "responses": ["Hmm, I'm not sure about that. Let's stick to book recommendations!","I am not programmed to answer these questions"]
    }
}

# Initialize your models and data
df = pd.read_csv("books_profile.csv")
embeddings_df = pd.read_csv("final_book_embeddings.csv")
embeddings = embeddings_df.iloc[:, 1:].values

EXCLUDED_TITLES = {"movie movie"}
STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'like', 'i', 'want'}

# Initialize Spell Checker
spell = SpellChecker(language='en', case_sensitive=False)
custom_book_titles = {title for title in df["title"]}
spell.word_frequency.load_words(custom_book_titles)

# Initialize NER Model
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    tokenizer="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# Add these functions before the chat_endpoint

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def detect_intent(user_input):
    cleaned_input = preprocess(user_input)
    for intent, data in INTENTS.items():
        if intent == "non_related":
            continue  # Handle last
        for pattern in data["patterns"]:
            if re.search(pattern, cleaned_input):
                return intent
    return "non_related"  # Default if no match

def generate_ngrams(words, max_n=5):
    """Generate n-grams from tokenized words."""
    ngrams = []
    for n in range(1, max_n+1):
        for i in range(len(words)-n+1):
            ngrams.append(' '.join(words[i:i+n]).lower())
    return ngrams

def correct_spelling(query):
    """Correct spelling while preserving title case."""
    words = query.split()
    corrected_words = []
    for word in words:
        if word.lower() in [t.lower() for t in custom_book_titles]:
            original_title = next(t for t in custom_book_titles if t.lower() == word.lower())
            corrected_words.append(original_title)
        else:
            corrected_word = spell.correction(word.lower())
            if word.istitle():
                corrected_word = corrected_word.title()
            elif word.isupper():
                corrected_word = corrected_word.upper()
            corrected_words.append(corrected_word)
    return ' '.join(corrected_words)

def extract_book_titles(query):
    # Step 1: Correct Spelling
    corrected_query = correct_spelling(query)
    
    # Step 2a: Extract NER Candidates (multi-word supported)
    results = ner_pipeline(corrected_query)
    ner_candidates = [
        entity["word"].lower() 
        for entity in results 
        if entity["entity_group"] in ["ORG", "MISC"]
    ]
    
    # Step 2b: Generate N-gram Candidates
    corrected_words = corrected_query.split()
    ngram_candidates = generate_ngrams(corrected_words)
    
    # Combine and deduplicate candidates
    all_candidates = list(set(ner_candidates + ngram_candidates))
    
    # Step 3: Context-Aware Filtering
    filtered_candidates = []
    for candidate in all_candidates:
        if ' ' in candidate:
            # Keep multi-word candidates (allow stop words)
            filtered_candidates.append(candidate)
        else:
            # Filter single-word candidates
            if len(candidate) > 2 and candidate not in STOP_WORDS:
                filtered_candidates.append(candidate)
    
    # Step 4: Validate with Fuzzy Matching
    title_lower_to_original = {title.lower(): title for title in df["title"]}
    BOOK_DATABASE = list(title_lower_to_original.keys())
    
    validated_titles = []
    for candidate in sorted(filtered_candidates, key=lambda x: -len(x)):
        match, score, _ = process.extractOne(
            candidate,
            BOOK_DATABASE,
            scorer=fuzz.WRatio
        )
        if score > 90 and match not in EXCLUDED_TITLES:
            validated_titles.append(title_lower_to_original[match])
            break
    
    return validated_titles[0] if validated_titles else ""

@app.route('/books_chatbot', methods=['POST'])
def chat_endpoint():
    data = request.json
    user_input = data.get('message')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    # Add spell correction
    corrected_input = correct_spelling(user_input)
    cleaned_input = preprocess(corrected_input)
    
    intent = detect_intent(cleaned_input)
    
    if intent == "greeting":
        response = random.choice(INTENTS[intent]["responses"])
    elif intent == "leaving":
        response = random.choice(INTENTS[intent]["responses"])
    elif intent == "services":
        response = random.choice(INTENTS[intent]["responses"])
    elif intent == "recommendation":
        book_title = extract_book_titles(corrected_input)
        if not book_title:
            response = "Could not identify a book title in your request."
        else:
            try:
                index = df[df['title'] == book_title].index[0]
                input_embedding = embeddings[index]
                
                # Compute cosine similarity
                similarities = cosine_similarity(input_embedding.reshape(1, -1), embeddings).flatten()
                
                # Get top 5 similar books (excluding the input itself)
                top_indices = similarities.argsort()[::-1][1:6]
                
                # Format the response
                similar_books = [df.iloc[idx]['title'] for idx in top_indices]
                response = {"book": book_title, "recommendations": similar_books}
                
            except IndexError:
                response = f"Title '{book_title}' not found in the dataset."
    else:  # non_related
        response = random.choice(INTENTS["non_related"]["responses"])
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
