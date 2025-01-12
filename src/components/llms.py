import torch
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    if provider == "huggingface":
        return _load_huggingface_chat_model(model)
    else:
        return init_chat_model(model, model_provider=provider)


def _load_huggingface_chat_model(model: str):
    from langchain_huggingface.chat_models import ChatHuggingFace
    from langchain_huggingface.llms import HuggingFacePipeline
    from transformers import BitsAndBytesConfig, pipeline

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
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
    chat = ChatHuggingFace(llm=llm)
    return chat
