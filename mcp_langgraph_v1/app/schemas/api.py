from pydantic import BaseModelfrom pydantic import Base_id: str
    final_answer: str
    trace: list[str]
    route: str | None = None



class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ChatResponse(BaseModel):
