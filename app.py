import streamlit as st
import re
import emoji
import nltk
from nltk.corpus import stopwords
import pandas as pd
from datasets import load_dataset

# Modeling
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Page config
st.set_page_config(
    page_title="AI Support Router", 
    page_icon="🤖", 
    layout="centered"
)

# NLTK setup
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

setup_nltk()
stop_words = set(stopwords.words('english'))
crucial_words = {'not', 'no', 'nor', 'but', 'however', 'although', 'very', 'never'}
final_stop_words = stop_words - crucial_words

contractions = {
    "didn't": "did not", "don't": "do not", "aren't": "are not",
    "can't": "cannot", "couldn't": "could not", "doesn't": "does not",
    "haven't": "have not", "isn't": "is not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "wasn't": "was not",
    "weren't": "were not", "i'm": "i am", "you're": "you are", "he's": "he is",
    "it's": "it is", "we're": "we are", "they're": "they are", "let's": "let us"
}

# --- TEXT CLEANING  ---
def clean_text_professional(text):
    if text in ["[deleted]", "[removed]"] or pd.isna(text):
        return ""
    text = text.lower()
    for word, replacement in contractions.items():
        text = text.replace(word, replacement)
    text = emoji.demojize(text)
    text = re.sub(r"[^a-z0-9\s:_]", "", text)
    words = text.split()
    filtered_words = [w for w in words if w not in final_stop_words]
    return " ".join(filtered_words)

# --- MODEL TRAINING & CACHING ---
@st.cache_resource(show_spinner="Downloading data and training model... (This takes a minute on first run)")
def load_and_train_model():
    # 1. Load Data
    dataset = load_dataset("go_emotions", "simplified")
    df_train = pd.DataFrame(dataset['train'])
    
    # 2. Map Labels
    emotion_names = dataset['train'].features['labels'].feature.names
    five_cat_map = {
        'anger': 'Angry', 'disgust': 'Angry', 'hate': 'Angry',
        'annoyance': 'Frustrated', 'disapproval': 'Frustrated', 'disappointment': 'Frustrated',
        'remorse': 'Frustrated', 'sadness': 'Frustrated', 'grief': 'Frustrated', 'embarrassment': 'Frustrated',
        'confusion': 'Confused', 'curiosity': 'Confused', 'realization': 'Confused',
        'surprise': 'Confused', 'nervousness': 'Confused', 'fear': 'Confused',
        'joy': 'Satisfied', 'excitement': 'Satisfied', 'pride': 'Satisfied',
        'admiration': 'Satisfied', 'gratitude': 'Satisfied', 'love': 'Satisfied',
        'relief': 'Satisfied', 'optimism': 'Satisfied', 'desire': 'Satisfied', 'amusement': 'Satisfied',
        'neutral': 'Calm', 'approval': 'Calm', 'caring': 'Calm'
    }

    def get_five_category_label(label_indices):
        current_emotions = [emotion_names[i] for i in label_indices]
        groups = [five_cat_map.get(e, 'Calm') for e in current_emotions]
        if 'Angry' in groups: return 'Angry'
        if 'Frustrated' in groups: return 'Frustrated'
        if 'Confused' in groups: return 'Confused'
        if 'Satisfied' in groups: return 'Satisfied'
        return 'Calm'

    df_train['category_5'] = df_train['labels'].apply(get_five_category_label)
    df_train['clean_text'] = df_train['text'].apply(clean_text_professional)
    df_train = df_train[df_train['clean_text'] != ""]

    # 3. Train fast Pipeline (TF-IDF + Oversampling + SGD)
    pipeline_sgd = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('sampler', RandomOverSampler(random_state=42)),
        ('clf', SGDClassifier(random_state=42, class_weight='balanced'))
    ])
    
    pipeline_sgd.fit(df_train['clean_text'], df_train['category_5'])
    return pipeline_sgd

# Initialize Model
model = load_and_train_model()

# --- ROUTING LOGIC ---
actions = {
    'Angry': "🔴 **URGENT:** Routing to Senior Agent (Risk Team). Please hold while I connect you to a manager.",
    'Frustrated': "🟠 **HIGH:** Routing to Retention Team. I understand your frustration, let me get a specialist to look into this.",
    'Confused': "🟣 **MEDIUM:** Routing to Support/Education. Let me pull up the right FAQ and connect you with a guide.",
    'Satisfied': "🟢 **LOW:** Auto-Reply. We're so glad you're happy! Let us know if you need anything else.",
    'Calm': "🔵 **STANDARD:** General Queue. I have logged your request and a standard agent will be with you shortly."
}

# --- UI FRONTEND ---
st.title("🤖 Intelligent Support Routing")
st.markdown("Type a customer message below to see how the Hybrid Emotion-Aware NLP routes the ticket.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am the automated routing bot. Please describe your issue, and I will route you to the appropriate team based on the urgency of your message."})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., I've been waiting for hours, this is ridiculous!"):
    # 1. Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

   
    
    # Business Rule 1: Is the customer shouting? (Mostly uppercase and long enough to be a sentence)
    is_shouting = prompt.isupper() and len(prompt) > 10
    
    # Business Rule 2: High-risk customer service keywords
    high_risk_keywords = ['money back', 'refund', 'sue', 'lawsuit', 'unacceptable', 'asap', 'ridiculous']
    contains_risk_word = any(word in prompt.lower() for word in high_risk_keywords)

    # Apply Hybrid Routing
    if is_shouting or contains_risk_word:
        prediction = 'Angry'
        trigger_reason = "*(Flagged by Business Rules: High-Risk Keyword or Shouting detected)*"
    else:
        # Fall back to the NLP model for nuanced text
        clean_prompt = clean_text_professional(prompt)
        # Fallback if stopwords remove everything
        if clean_prompt == "": 
            clean_prompt = prompt.lower()
            
        prediction = model.predict([clean_prompt])[0]
        trigger_reason = "*(Flagged by NLP Model)*"

    routing_response = actions.get(prediction, "Review required.")

    # 3. Display assistant response
    with st.chat_message("assistant"):
        st.markdown(f"**Detected Emotion:** *{prediction}* {trigger_reason}")
        st.markdown(routing_response)
        
    # 4. Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": f"**Detected Emotion:** *{prediction}* {trigger_reason}\n\n{routing_response}"
    })