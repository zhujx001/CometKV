
from .llama import LlamaModel
from .qwen import QwenModel
from .mistral import MistralModel
from transformers import AutoTokenizer
from .utils import add_model_args


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name, max_len, dtype, device, tokenizer=None):
    if 'Llama' in model_name:
        llm = LlamaModel(model_name,
                         max_length=max_len,
                         dtype=dtype,
                         device_map=device,
                         tokenizer=tokenizer)
    elif 'Qwen' in model_name:
        llm = QwenModel(model_name,
                        max_length=max_len,
                        dtype=dtype,
                        device_map=device,
                        tokenizer=tokenizer)
    elif 'Mistral' in model_name:
        llm = MistralModel(model_name,
                           max_length=max_len,
                           dtype=dtype,
                           device_map=device,
                           tokenizer=tokenizer)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm
