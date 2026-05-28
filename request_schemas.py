from pydantic import BaseModel


class QueryRequest(BaseModel):
    user_id: int
    query:   str