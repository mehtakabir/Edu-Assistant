import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm


def run_router_agent(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant for an educational system.
Classify the user query into exactly one category.

Categories:
- rag_query     : questions about study topics, notes, concepts from PDF documents
- student_query : questions about student info like attendance, marks, quiz status
- quiz_query    : requests to generate a quiz on a topic

Respond ONLY with JSON. No explanation. No markdown.
Format: {{"intent": "category_name"}}"""),
        ("human", "Classify: {query}")
    ])

    chain  = prompt | get_llm() | StrOutputParser()
    output = chain.invoke({"query": query})

    try:
        cleaned = re.sub(r"```(?:json)?|```", "", output).strip()
        result  = json.loads(cleaned)
        intent  = result.get("intent", "rag_query")
        valid   = {"rag_query", "student_query", "quiz_query"}
        return intent if intent in valid else "rag_query"
    except Exception:
        return "rag_query"