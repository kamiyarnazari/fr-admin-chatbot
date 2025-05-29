import json
from transformers import pipeline

# ─── CONFIG ───────────────────────────────────────────────────────────

# List the HF model IDs you want to compare:
MODEL_IDS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "bigscience/bloomz-1b7",
    "mosaicml/mpt-7b-instruct",          # << newly added
    "NousResearch/llama-2-7b-chat-hf"     # << newly added
]

# A handful of test questions (not in your FAQ) to probe the LLM:
QUESTIONS = [
    "Comment puis-je contacter la CAF pour une question sur mon dossier ?",
    "Je voudrais prendre rendez-vous avec la CPAM, comment procéder ?",
    "Quels sont les horaires d’ouverture de la préfecture ?",
    "Où télécharger le formulaire de déclaration d’impôts ?",
    "Peux-tu me raconter une blague en français ?"
]

# Device: 0 for first GPU, or -1 for CPU-only
DEVICE = 0  

# Decoding params—tweak these as you like
PIPE_KWARGS = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "max_new_tokens": 80,
    # Note: return_full_text must be stripped manually below
}

# ─── EVALUATION ────────────────────────────────────────────────────────

def main():
    all_results = {}

    for model_id in MODEL_IDS:
        print(f"\n Evaluating model: {model_id}")
        pipe = pipeline(
            "text2text-generation",
            model=model_id,
            tokenizer=model_id,
            device=DEVICE,
            **PIPE_KWARGS
        )

        model_outputs = []
        for q in QUESTIONS:
            prompt = (
                "Vous êtes un assistant administratif français. "
                "Ne répétez pas la question et répondez en phrases complètes.\n\n"
                f"Q: {q}\nA: "
            )
            full = pipe(prompt)[0]["generated_text"]
            # strip off the prompt prefix if echoed back
            answer = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
            print(f"Q: {q}\n→ {answer}\n")
            model_outputs.append(answer)

        all_results[model_id] = model_outputs

    # Write out for later inspection
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "questions": QUESTIONS,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)

    print("\n Done! See model_comparison_results.json for full outputs.")

if __name__ == "__main__":
    main()
