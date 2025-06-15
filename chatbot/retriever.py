import json
import re
import random
import os
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

"""
Hybrid semantic retriever and LLM-based generator for the French Administration Chatbot.

This module implements the `FAQWithLLM` class, which handles both:
1. **Semantic retrieval**: Finds the most relevant FAQ entries using multilingual sentence embeddings (SentenceTransformer).
2. **LLM fallback generation**: If no suitable match is found, generates a response using a fine-tuned LoRA adapter on top of the Mistral-7B model.

Key Features:
- Uses cosine similarity to retrieve top FAQ candidates
- Detects and responds to small talk (e.g., "bonjour", "ça va")
- Falls back to a quantized, fine-tuned LLM when no FAQ is confident enough
- Supports CPU and GPU with 4-bit quantization via `BitsAndBytesConfig`
- Loads LoRA adapters using `peft.PeftModel`

Usage:
- Called by the Gradio UI to handle user input and return the appropriate answer
- Returns either: (1) an exact or semantic match, or (2) an LLM-generated fallback

This architecture combines speed and control from FAQ with flexibility and coverage from LLMs.
"""




class FAQWithLLM:
    def __init__(self, faq_path: str):
        #Sentence‐Transformer FAQ retrieval setup
        self.model = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v1"
        )
        self.faq_data = self._load_faq(faq_path)
        self.questions = [item["question"] for item in self.faq_data]
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)

        self.small_talk_patterns = [
            r"\bbonjour\b", r"\bsalut\b", r"\bça va\b", r"\bcomment ça va\b"
        ]
        self.small_talk_replies = [
            "Bonjour ! Comment puis-je vous aider aujourd’hui ?",
            "Salut ! Tout va bien, et vous ?",
            "Je vais très bien, merci ! En quoi puis-je vous assister ?"
        ]

        #LoRA‐enabled LLM setup for fallback FAQ
        BASE_MODEL_ID    = "mistralai/Mistral-7B-v0.1"
        LORA_ADAPTER_DIR = "models/mistral-7b-finetuned"
        MAX_TOKENS       = 128

        #Loading tokenizer 
        print(f"\nLoading tokenizer from: {BASE_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        #Checking for GPU availability if not available fall back to CPU 
        if torch.cuda.is_available():
            device = "cuda"
            # Where to spill quantized layers that don't fit on GPU
            OFFLOAD_DIR = "offload_dir"
            os.makedirs(OFFLOAD_DIR, exist_ok=True)

            print(f"Loading base model (4-bit) on GPU with offload: {BASE_MODEL_ID}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                offload_folder=OFFLOAD_DIR,
                offload_state_dict=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto",           
                torch_dtype=torch.float16,
            )
            base_model.eval()

            print(f"Loading LoRA adapters (on GPU) from: {LORA_ADAPTER_DIR}")
            self.generator = PeftModel.from_pretrained(
                base_model,
                LORA_ADAPTER_DIR,
                torch_dtype=torch.float16
            )
            self.generator.to(device)
            self.device = device

        else:
        #No GPU available then fall back to CPU (FP32)
            device = "cpu"
            print(f" No GPU detected—loading base model (FP32) on CPU: {BASE_MODEL_ID}")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                device_map={"": "cpu"},
                torch_dtype=torch.float32
            )
            base_model.eval()

            print(f"📥 Loading LoRA adapters (CPU) from: {LORA_ADAPTER_DIR}")
            self.generator = PeftModel.from_pretrained(
                base_model,
                LORA_ADAPTER_DIR,
                torch_dtype=torch.float32
            )
            self.generator.to(device)
            self.device = device

        self.max_tokens = MAX_TOKENS

    def _load_faq(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_answer(self,
                   user_input: str,
                   chat_history: list[tuple[str, str]] = None,
                   threshold: float = 0.7,
                   top_k: int = 2,
                   high_confidence: float = 0.8) -> str:
        
        # If the user input is empty
        if not user_input.strip():
            return "Veuillez poser votre question et je vous aiderai avec plaisir."
        

        normalized = user_input.strip().lower()

        # Small‐talk detection
        for pat in self.small_talk_patterns:
            if re.search(pat, normalized):
                return random.choice(self.small_talk_replies)

        # Exact FAQ match
        for item in self.faq_data:
            if item["question"].strip().lower() == normalized:
                return item["answer"]

        # Semantic search via embeddings
        input_emb = self.model.encode(user_input, convert_to_tensor=True)
        scores = util.cos_sim(input_emb, self.embeddings)[0]
        top = scores.topk(k=top_k)
        top_indices = top.indices.tolist()
        top_scores  = top.values.tolist()

        # Super high‐confidence → return single FAQ answer
        if top_scores and top_scores[0] >= high_confidence:
            idx = top_indices[0]
            return self.faq_data[idx]["answer"]

        # Return up to top_k above threshold
        responses = []
        for idx, score in zip(top_indices, top_scores):
            if score < threshold:
                continue
            q = self.faq_data[idx]["question"]
            a = self.faq_data[idx]["answer"]
            responses.append(f"🔹 **{q}**\n{a}\n")

        # Fallback to LLM if no FAQ match
        if not responses:
            return self.generate_via_LLM(user_input, chat_history)

        return "\n".join(responses)

    def generate_via_LLM(self, user_input: str, chat_history=None) -> str:
        prompt = f"Réponds à la question administrative suivante :\n\n{user_input.strip()}\n\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.generator.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return full_text[len(prompt):].strip()
