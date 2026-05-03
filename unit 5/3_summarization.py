"""
===============================================================================
  NLP Unit 5 - Practice Program 3: TEXT SUMMARIZATION
===============================================================================
  Implements both Extractive and Abstractive summarization techniques.
  - Extractive: TF-IDF sentence scoring
  - Abstractive: Hugging Face Transformer pipeline (optional)
===============================================================================
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# 1. Sample Documents for Summarization
# ─────────────────────────────────────────────────────────────────────────────
DOCUMENTS = {
    "AI_Article": """
Artificial Intelligence (AI) has transformed numerous industries and continues 
to shape the future of technology. Machine learning, a subset of AI, enables 
computers to learn from data without explicit programming. Deep learning, which 
uses neural networks with many layers, has achieved remarkable results in image 
recognition, natural language processing, and game playing. The development of 
transformer architectures has revolutionized NLP tasks. Models like BERT and GPT 
have set new benchmarks in text understanding and generation. AI is being applied 
in healthcare for disease diagnosis, in finance for fraud detection, in 
transportation for autonomous vehicles, and in education for personalized learning. 
However, AI also raises ethical concerns about privacy, bias, and job displacement. 
Researchers are actively working on making AI systems more transparent, fair, and 
accountable. The future of AI depends on responsible development and deployment 
of these powerful technologies.
""",
    "Climate_Article": """
Climate change is one of the most pressing challenges facing humanity today. 
Global temperatures have risen by approximately 1.1 degrees Celsius since 
pre-industrial times. The burning of fossil fuels is the primary driver of 
greenhouse gas emissions. These emissions trap heat in the atmosphere, leading 
to rising temperatures, melting ice caps, and rising sea levels. Extreme weather 
events such as hurricanes, droughts, and floods are becoming more frequent and 
severe. The Paris Agreement aims to limit global warming to 1.5 degrees Celsius 
above pre-industrial levels. Countries around the world are implementing 
renewable energy solutions including solar, wind, and hydroelectric power. 
Carbon capture technologies are being developed to remove CO2 from the 
atmosphere. Individual actions such as reducing energy consumption, using public 
transportation, and eating less meat can also contribute to fighting climate 
change. Scientists emphasize that immediate and collective action is necessary 
to prevent the worst impacts of climate change.
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text):
    text = re.sub(r"\s+", " ", text.strip())
    return text


def split_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extractive Summarization using TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
def extractive_summarize_tfidf(text, num_sentences=3):
    """
    Extractive summarization by scoring sentences based on their
    TF-IDF similarity to the overall document.
    """
    clean_text = preprocess(text)
    sentences = split_sentences(clean_text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences), sentences, [1.0] * len(sentences)

    # Vectorize all sentences
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Compute document vector (mean of all sentence vectors)
    doc_vector = np.asarray(tfidf_matrix.mean(axis=0))

    # Score each sentence by similarity to document
    scores = []
    for i in range(len(sentences)):
        sim = cosine_similarity(tfidf_matrix[i], doc_vector).flatten()[0]
        scores.append(sim)

    scores = np.array(scores)

    # Select top sentences (maintain original order)
    top_indices = np.argsort(scores)[::-1][:num_sentences]
    top_indices_sorted = sorted(top_indices)

    summary_sentences = [sentences[i] for i in top_indices_sorted]
    summary = " ".join(summary_sentences)

    return summary, sentences, scores


# ─────────────────────────────────────────────────────────────────────────────
# 4. Extractive Summarization using TextRank (simplified)
# ─────────────────────────────────────────────────────────────────────────────
def extractive_summarize_textrank(text, num_sentences=3, damping=0.85, iterations=30):
    """
    TextRank-based extractive summarization.
    Builds a sentence similarity graph and ranks using PageRank-like algorithm.
    """
    clean_text = preprocess(text)
    sentences = split_sentences(clean_text)

    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    # Build similarity matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Normalize similarity matrix
    np.fill_diagonal(sim_matrix, 0)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_matrix = sim_matrix / row_sums

    # PageRank iteration
    n = len(sentences)
    scores = np.ones(n) / n
    for _ in range(iterations):
        scores = (1 - damping) / n + damping * norm_matrix.T.dot(scores)

    # Select top sentences
    top_indices = np.argsort(scores)[::-1][:num_sentences]
    top_indices_sorted = sorted(top_indices)
    summary = " ".join([sentences[i] for i in top_indices_sorted])
    return summary, scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_compression_ratio(original, summary):
    orig_words = len(original.split())
    summ_words = len(summary.split())
    return summ_words / orig_words if orig_words > 0 else 0


def simple_rouge_1(reference, hypothesis):
    """Compute simple ROUGE-1 (unigram overlap) F1 score."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    overlap = ref_tokens & hyp_tokens
    if not hyp_tokens or not ref_tokens:
        return 0.0
    precision = len(overlap) / len(hyp_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ─────────────────────────────────────────────────────────────────────────────
# 6. Demonstration / Results
# ─────────────────────────────────────────────────────────────────────────────
def run_demo():
    print("=" * 70)
    print("  NLP UNIT 5 — PRACTICE PROGRAM 3: TEXT SUMMARIZATION")
    print("  Extractive (TF-IDF + TextRank) Summarization")
    print("=" * 70)

    for doc_name, doc_text in DOCUMENTS.items():
        clean = preprocess(doc_text)
        print(f"\n{'─'*70}")
        print(f"  Document: {doc_name}")
        print(f"  Original Length: {len(clean.split())} words")
        print(f"{'─'*70}")
        print(f"\n  Original Text:\n  {clean[:300]}...\n")

        # Method 1: TF-IDF Extractive
        summary_tfidf, sentences, scores = extractive_summarize_tfidf(clean, num_sentences=3)
        ratio_tfidf = compute_compression_ratio(clean, summary_tfidf)

        print("  --- Method 1: TF-IDF Extractive Summarization ---")
        print(f"  Summary: {summary_tfidf[:250]}...")
        print(f"  Summary Length: {len(summary_tfidf.split())} words")
        print(f"  Compression Ratio: {ratio_tfidf:.2%}")
        print(f"\n  Sentence Scores:")
        for i, (sent, score) in enumerate(zip(sentences, scores)):
            marker = " ★" if score >= sorted(scores)[-3] else ""
            print(f"    [{score:.4f}]{marker} {sent[:70]}...")
        print()

        # Method 2: TextRank
        summary_tr, tr_scores = extractive_summarize_textrank(clean, num_sentences=3)
        ratio_tr = compute_compression_ratio(clean, summary_tr)

        print("  --- Method 2: TextRank Summarization ---")
        print(f"  Summary: {summary_tr[:250]}...")
        print(f"  Summary Length: {len(summary_tr.split())} words")
        print(f"  Compression Ratio: {ratio_tr:.2%}")

        # ROUGE-1 comparison
        rouge_score = simple_rouge_1(summary_tfidf, summary_tr)
        print(f"\n  ROUGE-1 F1 (TF-IDF vs TextRank): {rouge_score:.4f}")

    # Abstractive summarization
    print(f"\n{'─'*70}")
    print("  --- Abstractive Summarization (Hugging Face) ---")
    print(f"{'─'*70}\n")
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        text = preprocess(DOCUMENTS["AI_Article"])
        result = summarizer(text, max_length=80, min_length=25, do_sample=False)
        print(f"  Abstractive Summary: {result[0]['summary_text']}")
    except Exception as e:
        print(f"  [INFO] Abstractive summarization skipped: {e}")
        print("  Install with: pip install transformers torch")

    print("\n" + "=" * 70)
    print("  Program completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
