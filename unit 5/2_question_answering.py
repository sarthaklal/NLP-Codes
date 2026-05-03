"""
===============================================================================
  NLP Unit 5 - Practice Program 2: QUESTION ANSWERING
===============================================================================
  An extractive QA system using TF-IDF for retrieval and sentence extraction.
===============================================================================
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CONTEXTS = {
    "nlp": "Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language. NLP combines computational linguistics with statistical, machine learning, and deep learning models. Applications of NLP include machine translation, sentiment analysis, chatbots, text summarization, and speech recognition. NLP has evolved significantly with the introduction of transformer models like BERT and GPT.",
    "machine_learning": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models. Unsupervised learning finds hidden patterns in unlabeled data. Reinforcement learning trains agents to make sequences of decisions.",
    "deep_learning": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model complex patterns. Deep learning architectures include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for NLP tasks. The key advantage of deep learning is its ability to automatically learn feature representations from raw data. Deep learning has achieved state-of-the-art results in image recognition, NLP, speech recognition, and game playing.",
}


def preprocess_text(text):
    return re.sub(r"\s+", " ", text.strip())


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def retrieve_relevant_context(question, contexts):
    context_texts = [preprocess_text(v) for v in contexts.values()]
    context_keys = list(contexts.keys())
    corpus = context_texts + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    best_idx = np.argmax(similarities)
    return context_keys[best_idx], context_texts[best_idx], similarities[best_idx]


def extract_answer(question, context, top_k=2):
    sentences = split_into_sentences(context)
    if not sentences:
        return "No answer found.", []
    corpus = sentences + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(sentences[idx], similarities[idx]) for idx in top_indices]
    answer = " ".join([sentences[idx] for idx in sorted(top_indices)])
    return answer, results


def answer_question(question, contexts=CONTEXTS):
    topic, context, doc_score = retrieve_relevant_context(question, contexts)
    answer, sentence_scores = extract_answer(question, context)
    return {
        "question": question, "topic": topic,
        "doc_relevance": doc_score, "answer": answer,
        "sentence_scores": sentence_scores,
    }


def run_demo():
    print("=" * 70)
    print("  NLP UNIT 5 — PRACTICE PROGRAM 2: QUESTION ANSWERING")
    print("  TF-IDF Retrieval + Extractive QA System")
    print("=" * 70)

    test_questions = [
        "What is Natural Language Processing?",
        "What are the types of machine learning?",
        "What is deep learning used for?",
        "What are transformer models?",
        "How do neural networks learn features?",
        "What is supervised learning?",
        "What are CNNs used for?",
    ]

    print("\n--- TF-IDF Based Extractive QA ---\n")
    for question in test_questions:
        result = answer_question(question)
        print(f"  Q: {result['question']}")
        print(f"  Topic: {result['topic']} (relevance: {result['doc_relevance']:.4f})")
        answer_display = result['answer'][:150]
        print(f"  A: {answer_display}...")
        if result['sentence_scores']:
            print(f"  Top sentence score: {result['sentence_scores'][0][1]:.4f}")
        print("-" * 60)

    print("\n--- Detailed Analysis ---\n")
    question = "What are the applications of NLP?"
    result = answer_question(question)
    print(f"  Question: {question}")
    print(f"  Retrieved Topic: {result['topic']}")
    print(f"  Document Relevance Score: {result['doc_relevance']:.4f}\n")
    print("  Candidate Sentences (ranked by relevance):")
    for i, (sent, score) in enumerate(result['sentence_scores'], 1):
        display = sent[:80]
        print(f"    {i}. [{score:.4f}] {display}...")
    print(f"\n  Final Answer: {result['answer'][:200]}")

    # Transformer-based QA
    print("\n--- Transformer-Based QA (Hugging Face) ---\n")
    try:
        from transformers import pipeline
        qa_pipeline = pipeline("question-answering",
                               model="distilbert-base-cased-distilled-squad")
        context = preprocess_text(CONTEXTS["nlp"])
        for q in ["What is NLP?", "What are the applications of NLP?"]:
            r = qa_pipeline(question=q, context=context)
            print(f"  Q: {q}")
            print(f"  A: {r['answer']}  (score: {r['score']:.4f})")
            print("-" * 60)
    except Exception as e:
        print(f"  [INFO] Transformer QA skipped: {e}")
        print("  Install with: pip install transformers torch\n")

    print("\n" + "=" * 70)
    print("  Program completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
