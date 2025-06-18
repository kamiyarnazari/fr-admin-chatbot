import time
import json
from sentence_transformers import SentenceTransformer, util


"""
Benchmark script to compare semantic retrievers using a fixed test suite.

Evaluates accuracy and response time of various multilingual SentenceTransformer models
on a set of paraphrased and off-topic French administrative questions.

"""



# Loading the FAQ 
with open("data/faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]

# Define test suite
test_questions = [
      # — Exact matches (should return single answer) —
    ("Comment demander une carte vitale ?",                        0),
    ("Quels documents sont nécessaires pour le renouvellement d’un titre de séjour ?", 1),
    ("Comment faire une demande d'APL ?",                          2),
    ("Comment obtenir une attestation de droits ?",                3),

    # — Paraphrases of existing FAQs —
    ("Comment puis-je obtenir ma carte vitale ?",                 0),
    ("Où et comment récupérer ma carte vitale ?",                 0),
    ("Documents nécessaires pour prolonger mon titre de séjour ?",1),
    ("J’ai besoin des pièces pour renouveler mon titre de séjour",1),
    ("Je veux demander une aide au logement, comment procéder ?",  2),
    ("Je souhaite télécharger mon attestation de droits",         3),

    # — Tricky paraphrases —
    ("Procédure pour obtenir l'APL sur le site de la CAF ?",      2),
    ("Comment solliciter une aide personnalisée au logement ?",    2),
    ("Comment acquérir ma carte de sécurité sociale ?",            0),
    ("Processus pour renouveler son permis de séjour",            1),

    # — Off-topic / No-answer cases (should return None logic) —
    ("Quels sont les horaires d'ouverture de la piscine municipale ?", None),
    ("Meilleur restaurant à Paris ?",                            None),
    ("Quel temps fera-t-il demain ?",                            None),
    ("Traduction de 'hello' en français ?",                      None),
    ("Quelle est la capitale de l'Allemagne ?",                  None),
    ("Quel est le résultat du PSG aujourd'hui ?",                None),

    # — Other administrative but not in FAQ (None) —
    ("Comment obtenir un passeport ?",                           None),
    ("Comment déclarer mes impôts en ligne ?",                   None),
    ("Quelles démarches pour ouvrir un compte bancaire ?",       None),
    ("Comment faire une demande de permis de conduire ?",        None),

    # — Slight rephrasings (should still match) —
    ("Comment soumettre ma demande d'APL ?",                     2),
    ("Où trouver mon attestation de droits sur Ameli ?",         3),
]

# Benchmark function
def evaluate_model(model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions, convert_to_tensor=True)
    correct, total, times = 0, 0, []

    for user_q, correct_idx in test_questions:
        start = time.time()
        q_emb = model.encode(user_q, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, embeddings)[0]
        top_idx = scores.argmax().item()
        times.append(time.time() - start)

        total += 1
        if correct_idx is None:
            if scores[top_idx].item() < 0.3:
                correct += 1
        
        else:
            if top_idx == correct_idx:
                correct += 1

    accuracy = correct / total
    avg_time = sum(times) / len(times)
    print(f"{model_name}: accuracy = {accuracy:.2f}, avg_latency={avg_time*1000:.1f}ms")

# Running the benchmark
for m in [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "sentence-transformers/LaBSE"
]:
    evaluate_model(m)