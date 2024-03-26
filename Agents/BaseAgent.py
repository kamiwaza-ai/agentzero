import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel
from LLM.ChatProcessor import ChatProcessor
from AgentEvents.AgentEvents import AgentEvents

class BaseAgent(BaseModel):
    """
    BaseAgent serves as a base class for managing complex prompting strategies with language models.
    It supports chaining of prompts, logging of interactions, adjustment based on past experiences (memories),
    and mechanisms for evaluating and choosing among multiple outputs.
    """
    agent_id: str
    role: str = "BaseAgent"
    prompts = {}
    chat_processor: ChatProcessor
    agent_events: AgentEvents

    def __init__(self, bootstrap_servers: str, agent_id: str = None):
        """
        Initialize the BaseAgent with a language model, Kafka bootstrap servers, and an optional agent_id.
        If agent_id is not provided, a new UUID will be generated.
        """
        self.agent_id = agent_id if agent_id else str(uuid.uuid4())
        self.chat_processor = ChatProcessor(trace_id=self.agent_id)
        self.agent_events = AgentEvents(bootstrap_servers=bootstrap_servers)

    def load_prompts(self) -> None:
        """
        Method to load prompts from a file that matches the agent's role.
        The prompts are saved into self.prompts.
        """
        try:
            with open(f'prompts/{self.role}.json', 'r') as prompt_file:
                self.prompts = json.load(prompt_file)
        except FileNotFoundError:
            print(f"No prompts file found for role: {self.role}")
        except json.JSONDecodeError:
            print(f"Error decoding the prompts file for role: {self.role}")

    def select_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Select a model either passed as a parameter or available from our API endpoint.
        If the model is available, it makes a POST request to load the model.
        """
        import requests
        import json

        # If no model_name is provided, fetch the list of available models and select one (TBD)
        if not model_name:
            response = requests.get('http://firestorm.local:5000/v1/internal/model/list')
            if response.status_code == 200:
                model_list = response.json().get('model_names', [])
                preferred_models = [model for model in model_list if "mixtral*instruct*" in model or "dolphin*mixtral*" in model or "*deepseek-coder*" in model]
                if not preferred_models:
                    preferred_models = [model for model in model_list if "*gpt-3.5*" in model or "*gpt-4*" in model]
                    if "gpt-4-turbo" in model_list:
                        model_name = "gpt-4-turbo"
                    elif preferred_models:
                        model_name = preferred_models[0]  # Select the first match from the filtered list
                else:
                    model_name = preferred_models[0]  # Select the first match from the filtered list

                if not preferred_models and len(model_list) == 1:
                    model_name = model_list[0]  # If there's only one model, select it
                # If no model is selected by this point, an exception will be raised later

        # If a model_name is provided or selected, make a POST request to load the model
        model_params = {
            "*deepseek-coder*": {
                "args": {
                    "gpu_split": "15,24",
                    "trust-remote-code": True
                },
                "settings": {
                    "instruction_template": "DeepSeekCoder"
                }
            },
            "dolphin*mixtral*": {
                "args": {
                    # Add specific args for this model here
                },
                "settings": {
                    # Add specific settings for this model here
                }
            },
            "*mixtral*instruct": {
                "args": {
                    "tensorcores": True,
                    "tensor_split": "15,24",
                    "n-gpu-layers": "128",
                    "n_ctx": "32768",
                },
                "settings": {
                    "instruction_template": "Mistral"
                }
            }
        }

        if model_name:
            load_model_data = model_params.get(model_name, model_params["LoneStriker_deepseek-coder-33b-instruct-6.0bpw-h6-exl2"])
            response = requests.post('http://firestorm.local:5000/v1/internal/model/load', json=load_model_data)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Failed to load model: {model_name}")
        else:
            raise Exception("No model selected or available.")

    def think_on_objective(self, objective: str) -> Dict[str, Any]:
        """
        Method for the agent to think on its objective.
        """
        # TODO: Implement the method based on the objective
        pass

    def send_event(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Method for the agent to send an event to a Kafka topic.
        """
        self.agent_events.emit_event(topic, message)

    def recv_event(self, topic: str, group_id: str, timeout_ms: int = 10000) -> Dict[str, Any]:
        """
        Method for the agent to receive an event from a Kafka topic.
        """
        return next(self.agent_events.listen_for_events(topic, group_id, timeout_ms), None)
