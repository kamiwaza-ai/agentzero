from typing import Dict
from pydantic import BaseModel

# Define llm_params as a global variable
# The need/utility of these varies. The params here are
# largely the params exposed by oogabooga, including
# when the openai-compatible api is used; but others,
# like the llamacpp and vllm openai ones generally do not require
# these (in some cases they may require 'model' as a param, in others
# that may simply ignore it))

# If Kamiwaza is enabled, then by default we will also pull
# values from the Kamiwaza model config behind the deployment
# You can disable that behavior by setting the ChatProcessor
# instance variable
# chat_processor.kamiwaza_params = False
llm_params: Dict[str, Dict] = {
    "deepseek-coder": {
        "frequency_penalty": 0,
        "logit_bias": {},
        "max_tokens": 0,
        "n": 1,
        "presence_penalty": 0,
        "stream": True,
        "temperature": 1,
        "top_p": 0.2,
        "mode": "instruct",
        "continue_": False,
        "preset": "DeepSeekCoder",
        "min_p": 0,
        "top_k": 40,
        "instruction_template": "DeepSeek-Custom",
        "repetition_penalty": 1,
        "repetition_penalty_range": 1024,
        "typical_p": 1,
        "tfs": 1,
        "top_a": 0,
        "epsilon_cutoff": 0,
        "eta_cutoff": 0,
        "guidance_scale": 1,
        "negative_prompt": "",
        "penalty_alpha": 0,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "temperature_last": False,
        "do_sample": True,
        "seed": -1,
        "encoder_repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "min_length": 0,
        "num_beams": 1,
        "length_penalty": 1,
        "early_stopping": False,
        "truncation_length": 0,
        "max_tokens_second": 0,
        "custom_token_bans": "",
        "auto_max_new_tokens": True,
        "ban_eos_token": False,
        "add_bos_token": True,
        "skip_special_tokens": False,
        "grammar_string": "",
    },
    "phind": {
        "frequency_penalty": 0,
        "logit_bias": {},
        "max_tokens": 0,
        "n": 1,
        "presence_penalty": 0,
        "stream": True,
        "temperature": 1,
        "top_p": 0.2,
        "mode": "instruct",
        "continue_": False,
        "preset": "DeepSeekCoder",
        "min_p": 0,
        "top_k": 40,
        "instruction_template": "DeepSeek-Custom",
        "repetition_penalty": 1,
        "repetition_penalty_range": 1024,
        "typical_p": 1,
        "tfs": 1,
        "top_a": 0,
        "epsilon_cutoff": 0,
        "eta_cutoff": 0,
        "guidance_scale": 1,
        "negative_prompt": "",
        "penalty_alpha": 0,
        "mirostat_mode": 0,
        "mirostat_tau": 5,
        "mirostat_eta": 0.1,
        "temperature_last": False,
        "do_sample": True,
        "seed": -1,
        "encoder_repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "min_length": 0,
        "num_beams": 1,
        "length_penalty": 1,
        "early_stopping": False,
        "truncation_length": 0,
        "max_tokens_second": 0,
        "custom_token_bans": "",
        "auto_max_new_tokens": True,
        "ban_eos_token": False,
        "add_bos_token": True,
        "skip_special_tokens": False,
        "grammar_string": "",
    },
    "miqu": {
        "temperature": 0.1
    }
}

class LLMParams(BaseModel):
    """
    This class represents the parameters for the model.
    """

    @classmethod
    def get(cls, model_name: str) -> Dict:
        """
        This method returns the parameters for the given model name.
        """
        # Access the global variable llm_params directly
        return llm_params.get(model_name, {})
