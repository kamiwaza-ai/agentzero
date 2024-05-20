from fastapi import FastAPI, Request, WebSocket, APIRouter, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional
import json
import logging
import uuid
import os
from agentzero.LLM import ChatProcessor
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

chat_router = APIRouter()

logging.basicConfig(level=logging.DEBUG)

CHAT_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(CHAT_DIR, "templates"))

class ChatSession(BaseModel):
    chat_id: Optional[str]
    user_id: Optional[str]


# Consolidating route decorators for cleaner code
@chat_router.get('/chat/{chat_id}', response_class=HTMLResponse)
@chat_router.post('/chat/{chat_id}', response_class=HTMLResponse)
@chat_router.get('/chat', response_class=HTMLResponse, include_in_schema=False)
@chat_router.post('/chat', response_class=HTMLResponse, include_in_schema=False)
@chat_router.get('/', response_class=HTMLResponse, include_in_schema=False)
async def chat(request: Request, chat_id: Optional[str] = None, user_id: Optional[str] = Cookie(None)):
    """
    Handles chat requests, initializes chat sessions, and processes user input.
    
    Args:
        request (Request): The request object.
        chat_id (Optional[str], optional): The chat session ID. Defaults to None.
        user_id (Optional[str], optional): The user ID from cookies. Defaults to Cookie(None).
    
    Returns:
        Response: The HTML response for GET requests or JSON response for POST requests.
    """
    logging.debug(f"Requested URI: {request.url}")
    logging.debug(f"chat_id: {chat_id}")
    if not user_id:
        user_id = str(uuid.uuid4())

    user_data_dir = os.path.join(CHAT_DIR, f"userdata/{user_id}")
    os.makedirs(user_data_dir, exist_ok=True)

    last_model = None
    # Retrieve model selection details from the request
    if request.method == "POST":
        form_data = await request.form()
        host_name = form_data.get("host_name")
        listen_port = form_data.get("listen_port")
        model_name = form_data.get("model_selector")
        enable_retrieval = form_data.get("enable_retrieval") != 'False'  # Checkbox unchecked sends 'False'
    else:
        # Default values if not a POST request or values are missing
        host_name = "localhost"
        listen_port = "7777"
        model_name = None
        enable_retrieval = True  # Default to True if not a POST request or checkbox not present

    # Initialize ChatProcessor with model selection details and retrieval option
    processor = ChatProcessor(model=model_name, host_name=host_name, listen_port=listen_port, retrieval=enable_retrieval)

    if chat_id is None:
        logging.debug("No state, NEW SESSION")
        chat_id = str(uuid.uuid4())
        logging.debug(f"processor just created has messages: {processor.messages}")
        chat_history = []
        logging.debug(f"chat_history is: {chat_history}")
    else:
        chat_state_file = os.path.join(user_data_dir, f"{chat_id}.json")
        try:
            logging.debug(f"loading state from {chat_state_file}")
            processor.restore_state(file_path=chat_state_file)

            # if we had a saved model, and the user did not submit a model, then set that
            if processor.last_model and not model_name:
                last_model = processor.last_model
                processor.model = processor.last_model
            chat_history = processor.messages
        except FileNotFoundError as e:
            logging.error(f"Exception occurred restoring state: {e}")
            chat_history = []

    if request.method == "POST":
        user_input = form_data.get("userInput")
        chat_id = form_data.get("chatId")
        response = process_input(user_id, chat_id, user_input, host_name, listen_port, processor.model, enable_retrieval=enable_retrieval)

        return JSONResponse(content={"last_response": response['last_response']})

    context = {
        "request": request, 
        "user_id": user_id, 
        "chat_id": chat_id, 
        "chat_history": chat_history,
        "enable_retrieval": enable_retrieval  # Include retrieval status in context
    }
    if last_model is not None:
        context["selected_model"] = last_model
    
    response = templates.TemplateResponse("chat.html", context)
    response.set_cookie(key="user_id", value=user_id)

    return response

@chat_router.post('/init_chat', response_class=JSONResponse)
async def init_chat(request: Request, user_id: Optional[str] = Cookie(None)):
    if not user_id:
        user_id = str(uuid.uuid4())
    chat_id = str(uuid.uuid4())
    return JSONResponse(content={"user_id": user_id, "chat_id": chat_id})


@chat_router.get('/models', response_class=dict)
async def list_models():
    """
    Lists all available models by querying the model deployment service.
    Returns a list of models with their details including the listen port for ChatProcessor calls.

    Args:
        None

    Returns:
        dict: A dictionary containing a list of available models and their details.
    """
    import httpx
    from fastapi.responses import JSONResponse

    try:
        from kamiwaza.serving.services import ServingService
        ss = ServingService()
        model_deployments = ss.list_deployments(with_names=True)
        model_details = [
            {
                "model_id": str(getattr(md, "model_id", "default")),
                "host_name": str(getattr(md, "host_name", "localhost")),
                "model_name": str(getattr(md, "model_name", "default")),
                "listen_port": str(getattr(md, "listen_port", "8000")),
                "status": str(getattr(md, "status", "DEPLOYED")),
                "deployed_at": str(getattr(md, "deployed_at", "untracked"))
            } for md in model_deployments
        ]

        return JSONResponse(content={"models": model_details})
    except httpx.HTTPError as e:
        logging.error(f"Failed to fetch models: {e}")
        return JSONResponse(content={"error": "Failed to fetch models"}, status_code=500)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred"}, status_code=500)
async def process_input(user_id: str, chat_id: str, user_input: str, host_name: str, listen_port: str, model_name: str, stream_callback=None, enable_retrieval: bool = True) -> dict:
    """
    Processes the user input by generating a response using the ChatProcessor with updated parameters including model details.
    Restores the chat state from a file if it exists, and saves the updated state after processing.
    Adds the chat_id to the response if it exists.

    Args:
        user_id (str): The user's unique identifier.
        chat_id (str): The chat session's unique identifier.
        user_input (str): The input text from the user.
        host_name (str): The host address of the model server.
        listen_port (str): The port number of the model server.
        model_name (str): The name of the model to use for generating responses.
        stream_callback (callable, optional): A callback function for streaming data. Defaults to None.
        enable_retrieval (bool, optional): Whether to enable retrieval for the chat session. Defaults to True.

    Returns:
        dict: The response generated by the ChatProcessor, with chat_id added if it exists.
    """
    user_data_dir = os.path.join(CHAT_DIR, f"userdata/{user_id}")
    chat_state_file = os.path.join(user_data_dir, f"{chat_id}.json")
    processor = ChatProcessor(model=model_name, host_name=host_name, listen_port=int(listen_port), retrieval=enable_retrieval)
    try:
        processor.restore_state(file_path=chat_state_file)
    except FileNotFoundError as e:
        logging.error(f"Exception occurred restoring state: {e}")
    logging.debug("### process_input: generate response")
    response = await processor.generate_response(user_input, stream_callback=stream_callback)
    logging.debug(f"dumping state to {chat_state_file}")
    processor.dump_state(file_path=chat_state_file)
    
    # Add chat_id to the response if it exists
    if chat_id:
        response['chat_id'] = chat_id
    return response

@chat_router.websocket('/ws/{user_id}/{chat_id}/{host_name}/{listen_port}/{model_name}/{enable_retrieval}/')
async def websocket_endpoint(websocket: WebSocket, user_id: str, chat_id: str, host_name: str, listen_port: str, model_name: str, enable_retrieval: bool):
    await websocket.accept()

    logging.debug("GOOOOOO ####")
    while True:
        try:
            data = await websocket.receive_text()
            logging.debug(f"Received data from WebSocket: {data}")
            data_json = json.loads(data)
            user_input = data_json.get('user_input')

            if user_input:
                # Directly call process_input without passing a processor instance
                response = await process_input(
                    user_id=user_id, 
                    chat_id=chat_id, 
                    user_input=user_input, 
                    host_name=host_name, 
                    listen_port=listen_port, 
                    model_name=model_name, 
                    enable_retrieval=enable_retrieval
                )
                response_with_type = {"type": "response", **response}
                await websocket.send_text(json.dumps(response_with_type))
            else:
                logging.error("Received data does not contain 'user_input'")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from WebSocket: {e}")
            continue
        except WebSocketDisconnect as e:
            logging.info(f"WebSocket disconnected: {e}. Client likely closed the connection.")
            break
        except Exception as e:
            logging.error(f"Unmatched exception in websocket: {e}")
            raise e

@chat_router.get('/chats', response_class=JSONResponse)
async def chats(request: Request, user_id: Optional[str] = Cookie(None)):
    user_data_path = os.path.join(CHAT_DIR, f"userdata/{user_id}")
    chat_sessions = []

    if user_id and os.path.exists(user_data_path):
        titles_cache_path = os.path.join(user_data_path, 'titles.cache')
        titles_cache = {}
        if os.path.exists(titles_cache_path):
            with open(titles_cache_path, 'r') as f:
                titles_cache = json.load(f)
        logging.debug(os.listdir(user_data_path))
        for filename in os.listdir(user_data_path):
            if filename == 'titles.cache':
                continue
            logging.debug(f"listdir {filename}")
            if len(filename) > 2:
                guid = filename.split('.')[0]
                if guid not in titles_cache:
                    try:
                        with open(os.path.join(user_data_path, filename), 'r') as f:
                            data = json.load(f)
                            title = data.get("title", "Title Not Available")
                    except Exception as e:
                        title = "Title Not Available"
                        logging.error(f"Exception occurred loading title: {e}")
                    titles_cache[guid] = title

                chat_sessions.append({"chatId": guid, "title": titles_cache[guid]})

        with open(titles_cache_path, 'w') as f:
            valid_cache = {k: v for k, v in titles_cache.items() if os.path.exists(os.path.join(user_data_path, f"{k}.json"))}
            json.dump(valid_cache, f)

    return JSONResponse(content=chat_sessions)


@chat_router.post('/chat/{chat_id}/title', response_class=JSONResponse)
async def update_chat_title(request: Request, chat_id: str, user_id: Optional[str] = Cookie(None)):
    """
    Updates the title of a chat session with the specified chat_id for the user.

    Args:
        request (Request): The request object.
        chat_id (str): The chat session ID to update the title for.
        user_id (Optional[str], optional): The user ID from cookies. Defaults to None.

    Returns:
        JSONResponse: A JSON response indicating success or failure of the title update.
    """
    try:
        data = await request.json()
        new_title = data.get('title')
    except Exception as e:
        logging.error(f"Error extracting title from request data: {e}")
        return JSONResponse(content={"success": False, "message": "Invalid request data format"})

    user_data_path = os.path.join(CHAT_DIR, f"userdata/{user_id}")
    titles_cache_path = os.path.join(user_data_path, 'titles.cache')
    titles_cache = {}

    if os.path.exists(titles_cache_path):
        with open(titles_cache_path, 'r') as f:
            titles_cache = json.load(f)

        chat_file_path = os.path.join(user_data_path, f"{chat_id}.json")
        if os.path.exists(chat_file_path):
            try:
                with open(chat_file_path, 'r') as chat_file:
                    chat_data = json.load(chat_file)
                    chat_data["title"] = new_title
                with open(chat_file_path, 'w') as chat_file:
                    json.dump(chat_data, chat_file)

                titles_cache[chat_id] = new_title

                with open(titles_cache_path, 'w') as f:
                    json.dump(titles_cache, f)

                return JSONResponse(content={"success": True})
            except Exception as e:
                logging.error(f"Error updating chat title: {e}")
                return JSONResponse(content={"success": False})
        else:
            return JSONResponse(content={"success": False, "message": "Chat session not found"})
    else:
        return JSONResponse(content={"success": False, "message": "Titles cache not found"})



