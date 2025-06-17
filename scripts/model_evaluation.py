import mlflow
from sentence_transformers import SentenceTransformer, util
import json
import torch


mlflow.set_experiment("faq-semantic-vs-llm")

with mlflow.start_run():
    # Parameters
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
    threshold = 0.7
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("similarity_threshold", threshold)

    # Load model and FAQ
    model = SentenceTransformer(model_name)
    faq_data = json.load(open("data/faq.json", encoding="utf-8"))
    questions = [item["question"] for item in faq_data]
    embeddings = model.encode(questions, convert_to_tensor=True)

    # Evaluate 
    input_question = "Comment faire une demande de carte grise ?"
    input_emb = model.encode(input_question, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, embeddings)[0]

    top_score = torch.max(scores).item()
    mlflow.log_metric("top-similarity-score", top_score)

    if top_score < threshold:
        mlflow.log_metric("used-llm-fallback", 1)
    else:
        mlflow.log_metric("used-llm-fallback", 0)

    # log the generated responses
    results = {
        "top_score": top_score,
        "used_llm_fallback": top_score < threshold
    }

    with open("scripts/model-comparison-results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    mlflow.log_artifact("scripts/model-comparison-results.json", artifact_path = "evaluation")