import openai
import time
import types
import uuid
import logging
import json
import os
from typing import Dict, Optional, List
from pydantic import BaseModel
from agentzero.config import settings
from .ChatPrompts import chat_prompts

# Attempting to import external packages with error handling
import tiktoken

# we will use the full agentzero packages if available but if not, don't break

if settings.enable_events:
    try:
        from agentzero.AgentTools.AgentEvents.AgentEvents import AgentEvents
    except ImportError:
        AgentEvents = None
else:
    AgentEvents = None

try:
    from agentzero.LLM.LLMParams import LLMParams
except ImportError:
    class LLMParams:
        @staticmethod
        def get(model: str) -> Dict:
            return {}

try:
    from .ChatRetriever import ChatRetriever
except ImportError:
    ChatRetriever = None

# Set your API key as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY", "default_key")


class ChatProcessor:
    """
    ChatProcessor is a central do-it-all class for calling LLMs, maintaining a chat history, managing titles
    
    If you want to use it as a naive interface, you can pass stream=False, create_titles=False, retrieval=False and
    that takes it down to a more gentle wrapper.

    

    """

    TOKEN_THRESHOLD = 999999  # Token count threshold for distillation
    MSG_THRESHOLD = 200  # Minimum message token count to attempt distillation
    #   historically see 50-60% reduction but cycles, wait times, and our prompt itself is tokens.
    #   but you could adjust chat_processor.MSG_THRESHOLD manually to be aggressive in cases where, say,
    #   you know text will be re-used
    MAX_TOKENS = 16384  # Default maximum token count
    STREAM = True  # Flag to stream responses if a stream callback is provided
    CREATE_TITLES = True  # Flag to enable automatic title creation for chats
    MODEL = 'model'  # Default model identifier

    def __init__(self, model: str = MODEL, temperature: float = 0.0, trace_id: Optional[str] = None, host_name: str = "localhost", listen_port: int = 8000, model_id: Optional[str] = None, **kwargs):
        self.model = model
        self.title = "New Chat"
        self.temperature = temperature
        self.encoding_model = 'gpt-4' # gotta be a better way
        self.trace_id = trace_id or str(uuid.uuid4())
        self.logger = logging.getLogger(__name__)
        self.host_name = host_name
        self.listen_port = listen_port
        self.model_id = model_id
        # TODO: fix back to True
        self.retrieval = False
        self.kamiwaza_params = True
        self.reduced_messages = []
        self.last_response = None
        self.last_response_reason = None
        self.last_model = None

        # Handle the None case
        if not self.model:
            self.model = self.MODEL

        ## Optional kwargs
        if 'kamiwaza_params' in kwargs:
            self.kamiwaza_params = kwargs['kamiwaza_params']

        if 'create_titles' in kwargs:
            self.CREATE_TITLES = kwargs['create_titles']

        if 'stream' in kwargs:
            self.STREAM = kwargs['stream']

        if 'retrieval' in kwargs:
            self.retrieval = kwargs['retrieval']
            #TODO Fix back
            self.retrieval = False

        if not ChatRetriever:
            if self.retrieval:
                self.logger.debug("### ChatProcessor: retrieval was on, but ChatRetriever is not available, probably due to missing Kamiwaza libs; disabling")
            self.retrieval = False
            
        if self.retrieval:
            try:
                from kamiwaza.middleware.embedding.embedding_sentence_transformers import SentenceTransformerEmbedding
            except:
                self.retrieval = False
                self.logger.error("Failed to import kamiwaza.middleware.embedding.embedding_sentence_transformers.SentenceTransformerEmbedding; retrieval disabled")
                
        # Adjust logging level based on environment variable
        if os.getenv("KAMIWAZA_DEBUG_MODE", False):
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)


        # Set API base URL based on model specifics
        if not self.model.startswith('gpt-3') and not self.model.startswith('gpt-4'):
            # SSL verification flag
            if not self.host_name:
                self.host_name = "localhost"
            openai.verify_ssl_certs = False
            openai.api_base = f"http://{self.host_name}:{self.listen_port}/v1"

        # Adjust MAX_TOKENS based on model specifics
        self._adjust_max_tokens()

        self.reduction_enabled = False

        # Load LLM parameters for the model
        try:
            llm_params = LLMParams.get(self.model)
            self.logger.debug(f"LLM Params: {llm_params}")
        except Exception as e:
            logging.error(f"Failed to load LLM parameters for model {self.model}. Error: {str(e)}")
            llm_params = {}

        self.reducer_messages = None
        self.reducer_prompt = chat_prompts.message_distiller_prompt
        self.title_prompt = chat_prompts.title_creation_prompt

        if 'instruction_template' not in llm_params:
            self.start_messages = [
                {"role": "system", "content": chat_prompts.initial_system_message},
            ]
        else:
            self.start_messages = []
        self.messages = self.start_messages

    def _adjust_max_tokens(self):
        """Adjusts MAX_TOKENS based on the model specifics."""
        if self.model.startswith('gpt-4'):
            self.MAX_TOKENS = 8192
        elif self.model.startswith('gpt-4-32k'):
            if not os.getenv('ALLOW_GPT4_32K', None):
                self.logger.warn("### WARNING: gpt-4-32k is selected as the model. You PROBABLY don't want to use this; switch to gpt-4-turbo instead for longer context and much lower cost!")
            self.MAX_TOKENS = 32768
        elif self.model.startswith('gpt-4-t'):
            self.MAX_TOKENS = 131072
        elif 'gpt' not in self.model:
            # TODO: get away from relying on this
            self.MAX_TOKENS = int(self.MAX_TOKENS*0.96) # if we need to compute these a bit of padding for implementations that end up with more granular vocabs

    def strtokens(self, text):
        encoding = tiktoken.encoding_for_model(self.encoding_model)
        return 4 + len(encoding.encode(text))

    def tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculates the total number of tokens for a list of messages.

        Each message contributes a base number of tokens, and additional tokens are calculated
        based on the content of each message. A specific adjustment is made for messages with a 'name' key.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries.

        Returns:
            int: The total token count for the given messages.
        """
        token_count = 1  # Start with a base token count
        for message in messages:
            token_count += 4  # Base tokens for each message
            for key, value in message.items():
                token_count += self.strtokens(value)  # Tokens for the content
                if key == "name":
                    token_count -= 1  # Adjustment for 'name' key
        token_count += 2  # Final adjustment to token count
        return token_count

    # This is currently unused and questioning the interaction
    # with clients and streamed responses, so tbd
    # TODO: figure that out
    def call_chat_completion_with_retry(self, chat_params, retries=1):
        """
        Calls the OpenAI ChatCompletion.create API with basic retry logic and logs the time taken for the call.
        """
        try:
            start_time = time.time()  # Start time check
            response = openai.ChatCompletion.create(**chat_params)
            end_time = time.time()  # End time check
            self.logger.debug(f"ChatCompletion.create call successful in {end_time - start_time:.2f} seconds")
            return response, None  # Return response and no error
        except Exception as e:
            self.logger.error(f"Initial call to ChatCompletion.create failed: {str(e)}")
            if retries > 0:
                self.logger.info("Attempting retry...")
                try:
                    time.sleep(1)  # Simple backoff; consider exponential backoff for production use
                    start_time = time.time()  # Start time check for retry
                    response = openai.ChatCompletion.create(**chat_params)
                    end_time = time.time()  # End time check for retry
                    self.logger.debug(f"Retry of ChatCompletion.create call successful in {end_time - start_time:.2f} seconds")
                    return response, None  # Retry successful
                except Exception as retry_error:
                    self.logger.error(f"Retry call to ChatCompletion.create failed: {str(retry_error)}")
                    return None, retry_error  # Return no response and the error
            else:
                return None, e  # Return no response and the initial error

    def update_title(self):
        if not self.CREATE_TITLES or self.title != "New Chat":
            return

        self.logger.debug("Updating title")
        try:

            try:
                llm_params = LLMParams.get(self.model)
                self.logger.debug(f"LLM Params: {llm_params}")
            except Exception as e:
                logging.error(f"Failed to load LLM parameters for model {self.model}. Error: {str(e)}")
                llm_params = {}

            # Creating a dictionary with the parameters for the chat completion
            chat_params = {
                #'max_tokens': self.MAX_TOKENS,
                'messages': self.get_messages() + [{"role": "user", "content": self.title_prompt}]
            }

            # Merging the model parameters with the chat parameters
            chat_params.update({**llm_params, 'model': self.model})

            # Creating the chat completion with the merged parameters
            response = openai.ChatCompletion.create(**chat_params)
            self.logger.debug(str(response))
            if isinstance(response, types.GeneratorType):
                # Consume the generator and extract the string content from each OpenAIObject
                self.title = ''.join(chunk['choices'][0]['message']['content'] for chunk in response)
            else:
                self.title = response['choices'][0]['message']['content']
            self.logger.debug("Title updated to %s" % self.title)
        except Exception as e:
            # verbosely dump trace for debugging
            self.logger.debug(e, exc_info=True)
            self.title = "No Title available"
            
        return


    def reduce(self, model: str = MODEL, temperature: float = 0.1, trace_id: Optional[str] = None, host_name: str = "localhost", listen_port: int = 8000, model_id: Optional[str] = None, **kwargs):
        # Initialize reduced_count if not already set
        if not hasattr(self, 'reduced_count'):
            self.reduced_count = 0

        rm = self.reduced_messages

        # Iterate over messages starting from the last reduced message
        for role, message in self.messages:
            reduced_message = rm.pop(0) if rm else None
            if reduced_message:
                if 'role' not in reduced_message or reduced_message['role'] != message['role']:
                    self.logger.error(f"We tried to reduce {self.messages} but {self.reduced_messages} was out of sync; aborting reduction")
                    return
                # if we are here we have a message in reduced messages already
                continue
            if message["role"] == "system":
                # don't reduce a system message
                self.reduced_messages.append(message)
                continue

            mtokens = self.tokens([message])
            if mtokens < self.MSG_THRESHOLD:
                self.reduced_messages.append(message)
                continue

            self.logger.debug(f"Message {i} >= MSG_THRESHOLD with {mtokens} tokens, initiating distillation.")

            # Prepare the message for distillation without including the role
            distillation_prompt = chat_prompts.message_distiller_prompt + "\n\n" + m["content"]
            distillation_message = {
                "content": distillation_prompt
            }

            # Load LLM parameters for the model
            try:
                llm_params = LLMParams.get(self.model)
                self.logger.debug(f"LLM Params: {llm_params}")
            except Exception as e:
                logging.error(f"Failed to load LLM parameters for model {self.model}. Error: {str(e)}")
                llm_params = {}

            # Calculate tokens after adding distillation messages - maybe some day to monitor efficacy
            #tc = self.tokens([distillation_message])

            try:
                response = openai.ChatCompletion.create(
                    temperature=self.temperature,
                    model='gpt-3.5-turbo' if 'gpt' in self.model else self.model,
                    messages=[{"content": distillation_message}],  # Send only the content for distillation
                    # Incorporate model/api params from __init__
                    **llm_params,
                )
                distilled_content = response['choices'][0]['message']['content']
                self.reduced_messages.append({"role": message["role"], "content": distilled_content})
                self.logger.debug(f"Message {message['content']} distilled to {distilled_content}.")

            except Exception as e:
                self.logger.error(f"Failed to distill message {i}. Error: {str(e)}")
                self.reduced_messages.append(message)
                continue


    def get_messages(self) -> List[Dict[str, str]]:
        """
        Returns a list of messages, prioritizing reduced messages. If the count of reduced messages is less than the total messages,
        it fills the remainder with unreduced messages.

        :return: A list of message dictionaries.
        """
        # Calculate the number of additional messages needed from self.messages
        additional_messages_needed = len(self.messages) - len(self.reduced_messages)
        
        # Get the additional messages if needed
        additional_messages = self.messages[:additional_messages_needed] if additional_messages_needed > 0 else []
        
        # Combine reduced messages with the additional unreduced messages
        combined_messages = self.reduced_messages + additional_messages
        return combined_messages
        
        
    async def generate_response(self, prompt: str, stream_callback=None, chat_params=None) -> Dict[str, str]:
        """
        Generate a response from the chat model, logging input and output token counts as well as the total time taken.
        """
        self.last_response = None
        self.last_response_reason = None

        context_chunks = []
        self.logger.debug(f"generate_response called with retrieval {self.retrieval}, trace {self.trace_id} using {self.host_name} and {self.listen_port}")
        if self.retrieval:
            self.logger.debug(f"GOGO retrieve")
            retriever_processor = ChatProcessor(self.model,
                temperature = 0.1,
                trace_id = self.trace_id,
                host_name = self.host_name,
                listen_port = self.listen_port)
            cr = ChatRetriever(retriever_processor)
            
            context_chunks = await cr.get_relevant_chunks(prompt)
            self.logger.debug(f"Retrieved context chunks: {context_chunks}")

        self.logger.debug("log test")
        initial_token_count = self.tokens(self.get_messages())

        # Log the full system prompt, prompt, and response
        if AgentEvents:
            agent_events = AgentEvents()
        else:
            agent_events = None

        llm_params = LLMParams.get(self.model)

        prompt_to_send = prompt

        # if we are going to exceed the context length with all the chunks, strip them from the end until it fits
        if context_chunks:
            while initial_token_count + sum(self.tokens([{"user": chunk['data']}]) for chunk in context_chunks) > self.MAX_TOKENS and context_chunks:
                self.logger.debug(f"context chunk {context_chunks[-1]} cannot fit in context, removing")
                context_chunks.pop()

        # if we have content chunks from RAG, inject them
        if context_chunks:
            prompt_to_send = prompt_to_send + chat_prompts.rag_intro_prompt

            chunk_tpl = "Source: {source}, Offset: {offset}, Content below:\n{data}"
            for chunk in context_chunks:
                prompt_to_send = prompt_to_send + chunk_tpl.format(**chunk) + chat_prompts.rag_separator
        else:
            prompt_to_send = prompt

        # Ensure chat_params is a dictionary, initializing it if necessary
        if not isinstance(chat_params, dict):
            chat_params = {}

        # Update chat_params with messages, only set temperature if it's not already present
        chat_params.setdefault('temperature', self.temperature)
        chat_params['messages'] = self.get_messages() + [{"role": "user", "content": prompt_to_send}]

        # Merging the model parameters with the chat parameters
        chat_params.update({**llm_params, 'model': self.model})
        self.logger.debug(f"gogo still with {chat_params}")

        # Capture a start time for the request so we can debug timing
        start_time = time.time()

        if self.STREAM and stream_callback:
            self.last_response = ""
            if 'stream' not in chat_params:
                chat_params['stream'] = True
            response = openai.ChatCompletion.create(**chat_params)
            for chunk in response:
                # ignore chunks without content, which will be roles,
                # which for now can only be assistant in theory anyhow
                if chunk['choices'][0]['delta'].get('content'):
                    self.last_response = self.last_response + chunk['choices'][0]['delta']['content']
                    if stream_callback:
                        await stream_callback(chunk)
                    if chunk['choices'][0].get('finish_reason'):
                        self.last_response_reason = chunk['choices'][0]['finish_reason']

            self.messages.append({"role": "user", "content": prompt})
            self.messages.append(
                {"role": "assistant", "content": self.last_response})
        else:
            response = openai.ChatCompletion.create(
                **chat_params
            )
            self.last_response = response['choices'][0]['message']['content']
            self.last_response_reason = response['choices'][0]['finish_reason']
            self.messages.append({"role": "user", "content": prompt})
            self.messages.append(
                {"role": "assistant", "content": self.last_response})
            self.logger.debug(f"Got response from model: {self.last_response}")

        # Log the response
        self.logger.debug(f"ChatProcessor.generate_response to messages {self.messages}: {self.last_response}\n\nparams were {chat_params}")

        # Capture the end time
        end_time = time.time()

        # Calculate and output the duration of the call
        duration = end_time - start_time
        self.logger.debug(f"Duration of the call: {duration} seconds")

        final_token_count = self.tokens(self.messages)
        input_tokens = initial_token_count
        output_tokens = final_token_count - initial_token_count
        self.logger.debug(f"Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total tokens after generation: {final_token_count}")

        if agent_events:
            agent_events.emit_event("agent-logs", {"trace_id": self.trace_id, "messages": self.messages, "response": self.last_response})

        self.update_title()
        return {'last_response': self.last_response, 'chat_title': self.title, 'last_response_reason': self.last_response_reason}

    def dump_state(self, file_path: Optional[str] = None) -> None:
        """
        Dumps the current state of the chat, including messages, title, and model, to a specified file.

        Args:
            file_path (Optional[str]): The file path to dump the state to. Defaults to 'state.json' if not provided.
        """
        target_file = file_path or 'state.json'
        print(f"Dumping state to {target_file}")
        state_to_dump = {
            "messages": self.messages,
            "title": self.title,
            "model": self.model
        }
        try:
            with open(target_file, 'w', encoding='utf-8') as file:
                json.dump(state_to_dump, file)
        except Exception as e:
            print(f"Failed to dump state to {target_file}: {e}")

    def restore_state(self, file_path: Optional[str] = None) -> None:
        """
        Restores the state of the chat, including messages, title, and model, from a specified file.

        Args:
            file_path (Optional[str]): The file path to restore the state from. Defaults to 'state.json' if not provided.
        """
        target_file = file_path or 'state.json'
        if not os.path.exists(target_file):
            print(f"File {target_file} does not exist. State restoration aborted.")
            return
        try:
            with open(target_file, 'r', encoding='utf-8') as file:
                state = json.load(file)
                self.messages = state.get("messages", [])
                self.title = state.get("title", "New Chat")
                self.last_model = state.get("model", None)  # Restore model or default to class's MODEL attribute
        except Exception as e:
            print(f"Exception occurred while restoring state from {target_file}: {e}")

