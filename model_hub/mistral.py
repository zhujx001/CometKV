from transformers import MistralConfig, MistralForCausalLM

from .llama import LlamaModel


class MistralModel(LlamaModel):
    """
    Mistral uses the same projection and MLP layout as Llama in this runtime.
    """
    config_cls = MistralConfig
    hf_model_cls = MistralForCausalLM
