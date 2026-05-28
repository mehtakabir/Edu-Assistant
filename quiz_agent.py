from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_llm
from agents.rag_agent import get_context_for_topic, is_store_empty


def generate_quiz(topic: str) -> str:
    if is_store_empty():
        return "No study material found. Please upload PDF notes before generating a quiz."

    context = get_context_for_topic(topic)

    if not context:
        return f"No relevant content found for topic '{topic}' in the uploaded notes."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an educational assistant that creates quizzes strictly from study material.
Use ONLY the context below to generate questions.
Do NOT use any outside knowledge.

Context from study notes:
{context}"""),
        ("human", """Generate exactly 10 multiple choice questions on the topic: {topic}

Format the quiz EXACTLY like this — clean, structured, ready to print on A4 paper:

================================================================
                         QUIZ
                   Subject : Data Science
                   Total Marks: 10
================================================================

Q1. [Write the question here]

    A)  [Option A]
    B)  [Option B]
    C)  [Option C]
    D)  [Option D]


----------------------------------------------------------------

Q2. [Write the question here]

    A)  [Option A]
    B)  [Option B]
    C)  [Option C]
    D)  [Option D]


----------------------------------------------------------------

Follow this exact format for all 10 questions.
""")
    ])

    chain = prompt | get_llm() | StrOutputParser()

    try:
        return chain.invoke({"context": context, "topic": topic})
    except Exception as e:
        return f"Failed to generate quiz: {str(e)}"