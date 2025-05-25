import json
from sentence_transformers import SentenceTransformer, util

class FAQRetriever:
    def __init__(self, faq_path: str):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
        self.faq_data = self._load_faq(faq_path)
        self.questions = [item["question"] for item in self.faq_data]
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def _load_faq(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_answer(self, user_input: str, threshold: float = 0.3,
                    top_k: int = 2, high_confidence = 0.8) -> str:
        # If the question is the same in db just respond with the corresponding answer
        normalized = user_input.strip().lower()
        for item in self.faq_data:
            if item["question"].strip().lower() == normalized:
                return item["answer"]
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, self.embeddings)[0]

        # Debug print
        print(f"\n🔎 Similarity scores for: '{user_input}'")
        for i, s in enumerate(scores):
            print(f"{self.faq_data[i]['question']}: {s.item():.4f}")

        top_indices = scores.topk(k=top_k).indices.tolist()
        top_scores = scores.topk(k=top_k).values.tolist()

        # If the score is extremely high, then just return that one
        if top_scores[0] >= high_confidence:
            idx = top_indices[0]
            return self.faq_data[idx]["answer"]

        responses = []

        for idx, score in zip(top_indices, top_scores):
            if score < threshold:
                continue
            q = self.faq_data[idx]["question"]
            a = self.faq_data[idx]["answer"]
            responses.append(f"🔹 **{q}**\n{a}\n")

        if not responses:
            return "Je ne suis pas sûr de comprendre votre question. Pouvez-vous la reformuler ?"


        return "\n".join(responses)
    
