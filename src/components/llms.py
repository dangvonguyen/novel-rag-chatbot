import torch
import yaml
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig, pipeline


def load_chat_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    pipe = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_new_tokens=1024,
        temperature=0.3,
        return_full_text=False,
        device_map="auto",
        model_kwargs=dict(
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        ),
    )
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    llm = HuggingFacePipeline(pipeline=pipe)
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model