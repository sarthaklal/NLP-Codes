"""
===============================================================================
  NLP Unit 5 - Practice Program 1: CHATBOT
===============================================================================
  A rule-based + TF-IDF retrieval chatbot that can answer questions about
  common topics using cosine similarity for response selection.
===============================================================================
"""

import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# 1. Knowledge Base (corpus of documents the chatbot can reference)
# ─────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language.",
    "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
    "Deep learning is a subset of machine learning that uses neural networks with many layers to learn representations of data.",
    "A chatbot is a software application used to conduct an online chat conversation via text or text-to-speech.",
    "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    "TensorFlow and PyTorch are popular deep learning frameworks used for building neural networks.",
    "Transformers are a type of neural network architecture that has revolutionized NLP tasks like translation and summarization.",
    "BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model by Google.",
    "GPT (Generative Pre-trained Transformer) is a family of large language models developed by OpenAI.",
    "Tokenization is the process of breaking text into smaller units called tokens, such as words or subwords.",
    "Word embeddings are dense vector representations of words that capture semantic meaning.",
    "Sentiment analysis is the process of determining the emotional tone behind a series of words.",
    "Named Entity Recognition (NER) identifies and classifies named entities in text into predefined categories.",
    "Text classification is the task of assigning predefined categories to text documents.",
    "Recurrent Neural Networks (RNNs) are a class of neural networks designed for sequential data processing.",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Rule-Based Response Patterns (for greetings & small talk)
# ─────────────────────────────────────────────────────────────────────────────
RULE_PATTERNS = {
    r"\b(hi|hello|hey|greetings)\b": [
        "Hello! How can I help you today?",
        "Hi there! What would you like to know?",
        "Hey! I'm your NLP chatbot. Ask me anything!",
    ],
    r"\b(bye|goodbye|exit|quit)\b": [
        "Goodbye! Have a great day!",
        "See you later! Feel free to come back anytime.",
        "Bye! It was nice chatting with you.",
    ],
    r"\b(thanks|thank you|thankyou)\b": [
        "You're welcome!",
        "Happy to help!",
        "Glad I could assist!",
    ],
    r"\b(how are you|how do you do)\b": [
        "I'm just a chatbot, but I'm doing great! How can I help?",
        "I'm functioning perfectly! What can I do for you?",
    ],
    r"\b(your name|who are you)\b": [
        "I'm an NLP Practice Chatbot built for Unit 5!",
        "I'm a retrieval-based chatbot. Ask me about NLP topics!",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Text Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text):
    """Lowercase and remove special characters."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rule-Based Matching
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_response(user_input):
    """Check if the user input matches any rule-based pattern."""
    processed = preprocess(user_input)
    for pattern, responses in RULE_PATTERNS.items():
        if re.search(pattern, processed):
            return random.choice(responses)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. TF-IDF Retrieval-Based Response
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_based_response(user_input, threshold=0.15):
    """
    Use TF-IDF vectorization and cosine similarity to find the most
    relevant response from the knowledge base.
    """
    # Combine knowledge base with user query
    corpus = KNOWLEDGE_BASE + [user_input]

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Compute cosine similarity between user query and all KB entries
    user_vector = tfidf_matrix[-1]
    similarities = cosine_similarity(user_vector, tfidf_matrix[:-1]).flatten()

    # Get best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]

    if best_score >= threshold:
        return KNOWLEDGE_BASE[best_idx], best_score
    else:
        return None, best_score


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Chatbot Logic
# ─────────────────────────────────────────────────────────────────────────────
def chatbot_response(user_input):
    """Generate a response using rule-based or retrieval-based approach."""
    # Try rule-based first
    response = rule_based_response(user_input)
    if response:
        return response, "Rule-Based", 1.0

    # Fall back to retrieval-based
    response, score = retrieval_based_response(user_input)
    if response:
        return response, "TF-IDF Retrieval", score

    # Default fallback
    return (
        "I'm sorry, I don't have information about that. "
        "Try asking about NLP, machine learning, or deep learning!",
        "Fallback",
        0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Demonstration / Results
# ─────────────────────────────────────────────────────────────────────────────
def run_demo():
    print("=" * 70)
    print("  NLP UNIT 5 — PRACTICE PROGRAM 1: CHATBOT")
    print("  Rule-Based + TF-IDF Retrieval Chatbot")
    print("=" * 70)

    test_inputs = [
        "Hello!",
        "What is NLP?",
        "Tell me about deep learning",
        "What are transformers?",
        "Explain tokenization",
        "What is BERT?",
        "How are you?",
        "What is sentiment analysis?",
        "Tell me about word embeddings",
        "What is the weather today?",
        "Thank you!",
        "Goodbye",
    ]

    print("\n--- Chatbot Conversation Demo ---\n")
    for user_input in test_inputs:
        response, method, score = chatbot_response(user_input)
        print(f"  User   : {user_input}")
        print(f"  Bot    : {response}")
        print(f"  Method : {method} | Confidence: {score:.4f}")
        print("-" * 60)

    # Show TF-IDF similarity matrix for a sample query
    print("\n--- TF-IDF Similarity Analysis ---\n")
    sample_query = "What is natural language processing?"
    corpus = KNOWLEDGE_BASE + [sample_query]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    print(f"  Query: '{sample_query}'\n")
    print(f"  {'Rank':<6} {'Similarity':<12} {'Knowledge Base Entry'}")
    print(f"  {'─'*6} {'─'*12} {'─'*50}")
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    for rank, (idx, sim) in enumerate(ranked[:5], 1):
        entry = KNOWLEDGE_BASE[idx][:60] + "..." if len(KNOWLEDGE_BASE[idx]) > 60 else KNOWLEDGE_BASE[idx]
        print(f"  {rank:<6} {sim:<12.4f} {entry}")

    print("\n" + "=" * 70)
    print("  Program completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
