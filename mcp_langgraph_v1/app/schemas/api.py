from pydantic import BaseModel


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
    thread_id: str
    final_answer: str
    trace: list[str]
    route: str | None = None
