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
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            device_map = "auto",
            torch_dtype = "auto"
        )
        self.flan_pipe = pipeline(
             "text2text-generation",
             model=self.generator,
             tokenizer=self.tokenizer,
             max_length=200,
             do_sample=False,
             top_p=0.9,
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
            chat_history = None
    ) -> str:
        # Message array for message completion
        # messages = [
        #     {"role": "system", "content": "Vous êtes un assistant serviable pour des questions administratives en français."}
        # ]
        # if chat_history:
        #     for u, b in chat_history:
        #         messages.append({"role": "user",     "content": u})
        #         messages.append({"role:": "assistance", "content": b})

        # messages.append({"role": "user", "content": user_input})


        # try:
        #     # Calling the OpenAI API
        #     resp = client.chat.completions.create(
        #         model="gpt-3.5-turbo",
        #         messages=messages,
        #         max_tokens=200,
        #         temperature=0.7
        # )
        #     return resp.choices[0].message.content.strip()
        
        # except RateLimitError as e:
        #     print("OpenAI rate-limit:", e)  # logs the error
        #     return "⚠️ Le service IA est temporairement indisponible (quota). Voici quelques FAQ :\n\n" + self._faq_fallback(user_input)

        # except OpenAIError as e:
        #     # for other OpenAI issues (invalid key, timeout, etc.)
        #     print("OpenAI error:", e)
        #     # re-raise so you notice it—unless you want to fallback here too
        #     raise

        # except Exception as e:
        #     # unexpected errors
        #     print("Unexpected error in LLM call:", e)
        #     raise
        prompt = (
            "Vous êtes un assistant administratif français.\n"
            "Répondez aux questions marquées Q: par une réponse après A:.\n\n"
        )
        if chat_history:
             # chat_history is list of tuples (user,bot)
            turns = chat_history[-2:]  # last 2 pairs
            for u, b in turns:
                prompt += f"Q: {u}\nA: {b}\n"
            prompt += "\n"
        # Add the new question
        prompt += f"Q: {user_input}\nA:"

        # Running the LLM through local pipeline
        gen = self.flan_pipe(prompt, max_length=200)[0]["generated_text"]
        return gen.split("Q:", 1)[0].strip()
        
    def _faq_fallback(self, user_input: str) -> str:   
        """Simple two-answer fallback when LLM is unavailable."""
        input_emb = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(input_emb, self.embeddings)[0]
        top = scores.topk(k=2)
        resp = []
        for idx, score in zip(top.indices.tolist(), top.values.tolist()):
            if score < 0.3:
                continue
            item = self.faq_data[idx]
            resp.append(f"🔹 **{item['question']}**\n{item['answer']}")
        if not resp:
            return "*Je ne suis pas sûr de comprendre votre question. Pouvez-vous la reformuler ?*"
        return "\n\n".join(resp)
