from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)


SYSTEM_PROMPT = (
    "you will be asked questions by user about the novel: 2038"
    "You will answer the users question given the provided context"
    "the best that you can. If you do not know the answer,"
    "simply state: `I do not know`. Do not make anything up"
)
    

def generate(query: str, context_chunks: list[str]) -> str:
    if not context_chunks:
        return "I do not know"

    context = "\n".join([f"- {chunk}" for chunk in context_chunks])

    messages = [
        (
            "system",
            SYSTEM_PROMPT
        ),
        (
            "user",
            f"Context:\n\n{context}\n\nQuestion:\n{query}\n\nAnswer.",
        ),
    ]

    return llm.invoke(messages).content