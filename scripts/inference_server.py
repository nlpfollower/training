# scripts/inference_server.py
import pydevd_pycharm
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from typing import List
from src.coordination.model_node import ModelNode
from src.data.loader import RawDataset
from src.types.conversation import Chat, Role
from config import get_config, set_model_preset, update_config
from src.utils.logger import log

model_node = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_node, config
    config = get_config()
    set_model_preset(config, "llama3")
    config.model.max_sequence_length = 2048
    update_config(config, model_path="models/Meta-Llama-3.1-8B-Instruct")
    raw_dataset = RawDataset([""], "inference")
    model_node = ModelNode(config, raw_dataset, is_reference=True, debug=True)
    log.info("Model initialized and ready for inference")

    yield

    log.info("Shutting down the server")


app = FastAPI(lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    message: str


class InferenceRequest(BaseModel):
    conversation: List[ChatMessage]
    max_length: int = 100
    temperature: float = 0.6
    top_p: float = 0.9


@app.post("/generate")
async def generate(request: InferenceRequest):
    global model_node
    if model_node is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    conversation = [Chat(role=msg.role, message=msg.message) for msg in request.conversation]
    conversation.append(Chat(role=Role.assistant.value, message=""))
    updated_conversation = model_node.handle_inference_request(
        conversation,
        request.max_length,
        request.temperature,
        request.top_p
    )

    response = [ChatMessage(role=chat.role, message=chat.message) for chat in updated_conversation]
    log.info(f"Generated response: {response[-1].message}")
    return {"conversation": response}


if __name__ == "__main__":
    import uvicorn

    pydevd_pycharm.settrace('localhost', port=6789, stdoutToServer=True, stderrToServer=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)