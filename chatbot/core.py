import json
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re
import random



class FAQWithLLM:
    def __init__(self, faq_path: str):
        # High accuracy sentence embedding model choosen using the benchmark
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
        # Loading the FAQ data from faq.json and computing the embeddings
        self.faq_data = self._load_faq(faq_path)
        self.questions = [item["question"] for item in self.faq_data]
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

        # Small-talk pattern and canned replies
        self.small_talk_patterns = [
            r"\bbonjour\b", r"\bsalut\b", r"\bça va\b", r"\bcomment ça va\b"
        ]
        self.small_talk_replies = [
            "Bonjour ! Comment puis-je vous aider aujourd’hui ?",
            "Salut ! Tout va bien, et vous ?",
            "Je vais très bien, merci ! En quoi puis-je vous assister ?"
        ]

        # Loading Local LLm for fallback (Flan -T5 base)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            device_map = "auto",
            torch_dtype = "auto"
        )
        self.flan_pipe = pipeline(
             "text2text-generation",
             model=self.generator,
             tokenizer=self.tokenizer,
             max_length=200,
             do_sample=False         # greedy/beam search      
             )

    def _load_faq(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_answer(self,
                   user_input: str,
                   chat_history: list[tuple[str, str]] = None,
                   threshold: float = 0.7,
                   top_k: int = 2,
                   high_confidence = 0.8) -> str:
        '''Return either:
           1) small talk detection
           2) an FAQ answer (exact or high-confidence match), 
           3) up to top_k FAQ answers,
           4) or a generated LLM response if all else fails.'''
        
        normalized = user_input.strip().lower()

        # Detecting small-talks
        for pat in self.small_talk_patterns:
            if re.search(pat, normalized):
                return random.choice(self.small_talk_replies)
         
        # Exact match from FAQ
        for item in self.faq_data:
            if item["question"].strip().lower() == normalized:
                return item["answer"]
            
        # Embedding search
        input_embedding = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(input_embedding, self.embeddings)[0]
        top_indices = scores.topk(k=top_k).indices.tolist()
        top_scores = scores.topk(k=top_k).values.tolist()

        # High confidence single answer 
        if top_scores[0] >= high_confidence:
            idx = top_indices[0]
            return self.faq_data[idx]["answer"]

        # Returning up to top_k FAQ answers above the threshold
        responses = []
        for idx, score in zip(top_indices, top_scores):
            if score < threshold:
                continue
            q = self.faq_data[idx]["question"]
            a = self.faq_data[idx]["answer"]
            responses.append(f"🔹 **{q}**\n{a}\n")

        # No suitable FAQ match -> use LLM Fallback
        if not responses:
            return self.generate_via_LLM(user_input, chat_history)

        return "\n".join(responses)
    
    
    
    def generate_via_LLM(
            self,
            user_input: str,
            chat_history
    ) -> str:
        prompt = []
        if chat_history:
             # chat_history is list of tuples (user,bot)
            turns = chat_history[-4:]  # last 4 pairs (2 Q/As)
            for msg in turns:
                if msg["role"] == "user":
                    prompt += f"Q: {msg['content']}\n"
                else:
                    prompt += f"A: {msg['content']}\n"
            prompt += "\n"
        #Add the new question
        prompt = f"Q: {user_input}\nA:"

        # Running the LLM through local pipeline
        gen = self.flan_pipe(
            prompt,
            max_length=100
            )[0]["generated_text"]
        
        return gen.strip()
