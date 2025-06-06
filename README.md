#French Administrative FAQ Chatbot with LoRA-tuned Mistral-7B
##This project delivers a bilingual AI assistant tailored to answer French administrative questions using a hybrid approach that combines:

Semantic FAQ retrieval via multilingual sentence embeddings, and

Natural language generation using a fine-tuned Mistral-7B-Instruct model with LoRA adapters.

The assistant is able to:

1.Recognize and reply to predefined FAQs with high accuracy.

2.Handle rephrased or semantically similar queries using cosine similarity on multilingual embeddings.

3.Fall back to generative answers via a LoRA-finetuned large language model when no close match exists.

4.Detect and respond appropriately to small-talk or greeting messages.

##**Technical Stack**
Base Model: Mistral-7B-Instruct-v0.1

Fine-Tuning Method: Parameter-Efficient Fine-Tuning with LoRA

Retrieval: Sentence Transformers (distiluse-base-multilingual-cased-v1)

Generation: Transformers with GPU or CPU support

Quantization: 4-bit inference via bitsandbytes (when GPU is available)

Fallback: Full CPU-based FP32 inference for low-resource machines

##**Features**
✅ Supports GPU and CPU environments with automatic switching

✅ 4-bit inference with bitsandbytes for memory efficiency on GPU

✅ Fully localized for French administrative language

✅ Modular and production-ready structure (core.py, faq.json, model folders)

✅ Easily extensible with new questions and fine-tuning data

##**Use Cases**
Public service chatbots (CAF, CPAM, URSSAF, etc.)

Local administration helpdesks

Language model fine-tuning pipelines (LoRA/QLoRA)

French NLP/NLU research and experimentation