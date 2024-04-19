import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agentzero.Modules.Chat.app import chat_router  # Import chat_router from the chat module
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Define a list of origins that are allowed to make requests to this API
origins = [
    "*"
]

# Include the chat_router to make the chat module's routes accessible via the FastAPI app
app.include_router(chat_router)


# Mount the static directory to serve static files
app.mount("/chat/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__),"Modules/Chat/static")), name="static")


# Add CORS middleware to allow cross-origin requests from the allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)


