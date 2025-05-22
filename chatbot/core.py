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

    def get_answer(self, user_input: str) -> str:
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, self.embeddings)[0]
        best_match_idx = scores.argmax().item()
        return self.faq_data[best_match_idx]["answer"]