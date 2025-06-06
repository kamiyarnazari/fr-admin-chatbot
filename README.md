# French Administrative FAQ Chatbot with LoRA-tuned Mistral-7B

## Overview

This project delivers a bilingual AI assistant tailored to answer French administrative questions using a hybrid approach that combines:

- **Semantic FAQ retrieval** via multilingual sentence embeddings  
- **Natural language generation** using a fine-tuned Mistral-7B-Instruct model with LoRA adapters

### Capabilities

1. Recognize and reply to predefined FAQs with high accuracy  
2. Handle rephrased or semantically similar queries using cosine similarity  
3. Fall back to generative answers via a LoRA-tuned language model  
4. Detect and respond to small-talk or greetings

## Technical Stack

- **Base Model**: Mistral-7B-Instruct-v0.1  
- **Fine-Tuning**: LoRA (Low-Rank Adaptation)  
- **Retrieval**: Sentence Transformers (`distiluse-base-multilingual-cased-v1`)  
- **Generation**: Transformers with CPU/GPU fallback  
- **Quantization**: 4-bit inference via `bitsandbytes` (GPU only)  
- **Fallback**: Full FP32 CPU inference for low-resource machines

## Features

- ✅ GPU & CPU support with auto-detection  
- ✅ Efficient 4-bit quantized generation on GPU  
- ✅ Fully localized for French administrative content  
- ✅ Modular codebase: `core.py`, `faq.json`, adapter folders  
- ✅ Easily extensible and production-ready

## Use Cases

- Public service virtual agents (CAF, CPAM, URSSAF)  
- Local government helpdesks  
- Fine-tuning pipelines (LoRA/QLoRA)  
- French NLP/NLU experimentation
