"""
===============================================================================
  NLP Unit 5 - Practice Program 4: MACHINE TRANSLATION
===============================================================================
  Implements a simple word-level statistical translation system and
  demonstrates Seq2Seq concepts for English-to-French translation.
===============================================================================
"""

import re
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 1. Parallel Corpus (English-French sentence pairs)
# ─────────────────────────────────────────────────────────────────────────────
PARALLEL_CORPUS = [
    ("hello", "bonjour"),
    ("good morning", "bonjour"),
    ("how are you", "comment allez vous"),
    ("thank you", "merci"),
    ("goodbye", "au revoir"),
    ("yes", "oui"),
    ("no", "non"),
    ("please", "s il vous plait"),
    ("i am a student", "je suis un etudiant"),
    ("i love programming", "j aime la programmation"),
    ("the cat is on the table", "le chat est sur la table"),
    ("he is a good teacher", "il est un bon professeur"),
    ("she reads a book", "elle lit un livre"),
    ("we are learning french", "nous apprenons le francais"),
    ("the weather is nice today", "le temps est beau aujourd hui"),
    ("i like to eat pizza", "j aime manger de la pizza"),
    ("the dog is running", "le chien court"),
    ("they are playing football", "ils jouent au football"),
    ("this is a beautiful house", "c est une belle maison"),
    ("i want to learn machine learning", "je veux apprendre l apprentissage automatique"),
    ("natural language processing is interesting", "le traitement du langage naturel est interessant"),
    ("the book is on the table", "le livre est sur la table"),
    ("i am happy", "je suis heureux"),
    ("she is a doctor", "elle est medecin"),
    ("we love music", "nous aimons la musique"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text):
    return re.sub(r"[^a-z\s']", "", text.lower().strip())


def tokenize(text):
    return preprocess(text).split()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Word-Level Translation Model (IBM Model 1 — Simplified)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleTranslationModel:
    """
    A simplified IBM Model 1 implementation for word alignment
    and translation probabilities using EM algorithm.
    """

    def __init__(self):
        self.trans_prob = defaultdict(lambda: defaultdict(float))
        self.word_dict = {}

    def train(self, parallel_corpus, iterations=10):
        """Train translation probabilities using EM algorithm."""
        # Initialize uniform probabilities
        target_vocab = set()
        for src, tgt in parallel_corpus:
            for w in tokenize(tgt):
                target_vocab.add(w)

        n_target = len(target_vocab)
        for src, tgt in parallel_corpus:
            src_tokens = tokenize(src)
            tgt_tokens = tokenize(tgt)
            for s in src_tokens:
                for t in tgt_tokens:
                    self.trans_prob[s][t] = 1.0 / n_target

        # EM iterations
        for iteration in range(iterations):
            counts = defaultdict(lambda: defaultdict(float))
            totals = defaultdict(float)

            for src, tgt in parallel_corpus:
                src_tokens = tokenize(src)
                tgt_tokens = tokenize(tgt)

                for s in src_tokens:
                    z = sum(self.trans_prob[s][t] for t in tgt_tokens)
                    for t in tgt_tokens:
                        c = self.trans_prob[s][t] / z if z > 0 else 0
                        counts[s][t] += c
                        totals[s] += c

            # Update probabilities
            for s in counts:
                for t in counts[s]:
                    self.trans_prob[s][t] = counts[s][t] / totals[s] if totals[s] > 0 else 0

        # Build word dictionary (best translation for each word)
        for s in self.trans_prob:
            best_t = max(self.trans_prob[s], key=self.trans_prob[s].get)
            self.word_dict[s] = (best_t, self.trans_prob[s][best_t])

    def translate_word(self, word):
        """Translate a single word."""
        word = word.lower()
        if word in self.word_dict:
            return self.word_dict[word]
        return (word, 0.0)  # unknown word

    def translate_sentence(self, sentence):
        """Translate a sentence word by word."""
        tokens = tokenize(sentence)
        translated = []
        details = []
        for token in tokens:
            trans, prob = self.translate_word(token)
            translated.append(trans)
            details.append((token, trans, prob))
        return " ".join(translated), details

    def get_top_translations(self, word, top_k=5):
        """Get top-k translation candidates for a word."""
        word = word.lower()
        if word not in self.trans_prob:
            return []
        probs = self.trans_prob[word]
        sorted_trans = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_trans[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 4. BLEU Score (simplified unigram)
# ─────────────────────────────────────────────────────────────────────────────
def simple_bleu(reference, hypothesis):
    """Compute simplified unigram BLEU score."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0
    ref_counts = defaultdict(int)
    for t in ref_tokens:
        ref_counts[t] += 1
    matches = 0
    hyp_counts = defaultdict(int)
    for t in hyp_tokens:
        hyp_counts[t] += 1
    for t in hyp_counts:
        matches += min(hyp_counts[t], ref_counts.get(t, 0))
    precision = matches / len(hyp_tokens)
    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / len(ref_tokens)) if ref_tokens else 0
    return bp * precision


# ─────────────────────────────────────────────────────────────────────────────
# 5. Seq2Seq Concept Demonstration
# ─────────────────────────────────────────────────────────────────────────────
def demonstrate_seq2seq_concept():
    """Demonstrate Seq2Seq architecture concepts for MT."""
    print("\n  --- Seq2Seq Architecture Concepts ---\n")
    print("  Encoder-Decoder Architecture for Machine Translation:")
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  INPUT: 'I love programming'                       │")
    print("  │                                                     │")
    print("  │  ENCODER (processes source language):               │")
    print("  │  [I] → [love] → [programming]                      │")
    print("  │   h1  →  h2   →   h3 (context vector)              │")
    print("  │                                                     │")
    print("  │  CONTEXT VECTOR: h3 (fixed-size representation)     │")
    print("  │                                                     │")
    print("  │  DECODER (generates target language):               │")
    print("  │  <SOS> → [j'] → [aime] → [la] → [programmation]    │")
    print("  │                                                     │")
    print("  │  OUTPUT: 'j aime la programmation'                  │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Key Components:")
    print("  • Encoder RNN/LSTM: Reads source sentence, produces context")
    print("  • Context Vector: Compressed representation of source")
    print("  • Decoder RNN/LSTM: Generates target sentence word by word")
    print("  • Attention Mechanism: Allows decoder to focus on relevant")
    print("    parts of the source sentence at each decoding step")
    print("  • Transformer: Replaces RNNs with self-attention for")
    print("    parallel processing and better long-range dependencies")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Demonstration / Results
# ─────────────────────────────────────────────────────────────────────────────
def run_demo():
    print("=" * 70)
    print("  NLP UNIT 5 — PRACTICE PROGRAM 4: MACHINE TRANSLATION")
    print("  Statistical Word-Level Translation (IBM Model 1)")
    print("=" * 70)

    # Train the model
    print("\n--- Training Translation Model ---\n")
    model = SimpleTranslationModel()
    model.train(PARALLEL_CORPUS, iterations=15)
    print(f"  Training corpus: {len(PARALLEL_CORPUS)} sentence pairs")
    print(f"  Vocabulary size: {len(model.word_dict)} source words")
    print("  Training completed (15 EM iterations).")

    # Show learned word translations
    print("\n--- Learned Word Translation Table ---\n")
    print(f"  {'English':<20} {'French':<20} {'Probability':<12}")
    print(f"  {'─'*20} {'─'*20} {'─'*12}")
    sample_words = ["i", "the", "is", "love", "learning", "good",
                    "am", "book", "cat", "dog", "happy", "we"]
    for word in sample_words:
        if word in model.word_dict:
            trans, prob = model.word_dict[word]
            print(f"  {word:<20} {trans:<20} {prob:<12.4f}")

    # Show top translation candidates
    print("\n--- Top Translation Candidates ---\n")
    for word in ["the", "is", "i"]:
        candidates = model.get_top_translations(word, top_k=5)
        cand_str = ", ".join([f"{t}({p:.3f})" for t, p in candidates])
        print(f"  '{word}' → {cand_str}")

    # Translate test sentences
    print("\n--- Translation Results ---\n")
    test_sentences = [
        ("i am happy", "je suis heureux"),
        ("the cat is on the table", "le chat est sur la table"),
        ("i love programming", "j aime la programmation"),
        ("she is a doctor", "elle est medecin"),
        ("we love music", "nous aimons la musique"),
        ("good morning", "bonjour"),
    ]

    print(f"  {'Source (English)':<30} {'Translation':<30} {'Reference':<30} {'BLEU'}")
    print(f"  {'─'*30} {'─'*30} {'─'*30} {'─'*6}")
    for src, ref in test_sentences:
        translation, details = model.translate_sentence(src)
        bleu = simple_bleu(ref, translation)
        print(f"  {src:<30} {translation:<30} {ref:<30} {bleu:.3f}")

    # Detailed word-by-word translation
    print("\n--- Detailed Word-by-Word Translation ---\n")
    test_sent = "i love programming"
    translation, details = model.translate_sentence(test_sent)
    print(f"  Source: '{test_sent}'")
    print(f"  Translation: '{translation}'\n")
    print(f"  {'Source Word':<15} {'Target Word':<15} {'Confidence':<12}")
    print(f"  {'─'*15} {'─'*15} {'─'*12}")
    for src_w, tgt_w, prob in details:
        print(f"  {src_w:<15} {tgt_w:<15} {prob:<12.4f}")

    # BLEU score analysis
    print("\n--- BLEU Score Analysis ---\n")
    total_bleu = 0
    count = 0
    for src, ref in test_sentences:
        trans, _ = model.translate_sentence(src)
        bleu = simple_bleu(ref, trans)
        total_bleu += bleu
        count += 1
    avg_bleu = total_bleu / count if count > 0 else 0
    print(f"  Average BLEU Score: {avg_bleu:.4f}")
    print(f"  Total test sentences: {count}")

    # Seq2Seq concepts
    demonstrate_seq2seq_concept()

    # Transformer-based translation
    print("\n--- Transformer-Based Translation (Hugging Face) ---\n")
    try:
        from transformers import pipeline
        translator = pipeline("translation_en_to_fr",
                              model="Helsinki-NLP/opus-mt-en-fr")
        test = ["I love programming.", "The weather is nice today.",
                "Machine learning is interesting."]
        for t in test:
            result = translator(t)
            print(f"  EN: {t}")
            print(f"  FR: {result[0]['translation_text']}")
            print("-" * 50)
    except Exception as e:
        print(f"  [INFO] Transformer translation skipped: {e}")
        print("  Install: pip install transformers torch sentencepiece")

    print("\n" + "=" * 70)
    print("  Program completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
