import json
from sentence_transformers import SentenceTransformer, util

class FAQRetriever:
    def __init__(self, faq_path: str):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.faq_data = self._load_faq(faq_path)
        self.questions = [item["question"] for item in self.faq_data]
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

    def _load_faq(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_answer(self, user_input: str, threshold: float = 0.3, top_k: int = 2) -> str:
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, self.embeddings)[0]

        # Debug print
        print(f"\n🔎 Similarity scores for: '{user_input}'")
        for i, s in enumerate(scores):
            print(f"{self.faq_data[i]['question']}: {s.item():.4f}")

        top_indices = scores.topk(k=top_k).indices.tolist()
        top_scores = scores.topk(k=top_k).values.tolist()

        responses = []

        for idx, score in zip(top_indices, top_scores):
            if score < threshold:
                continue
            q = self.faq_data[idx]["question"]
            a = self.faq_data[idx]["answer"]
            responses.append(f"🔹 **{q}**\n{a}\n")
        # best_match_idx = scores.argmax().item()
        # best_score = scores[best_match_idx].item()

        if not responses:
            return "Je ne suis pas sûr de comprendre votre question. Pouvez-vous la reformuler ?"
        
        # matched_question = self.faq_data[best_match_idx]["question"]
        # answer = self.faq_data[best_match_idx]["answer"]

        return "\n".join(responses)
    
