# Step 1 of the pipeline: LLM based intent classification, then the router
# sends EXPLAIN to the no-RAG path and GENERATE to the RAG path.

from llm import get_llm
from prompts import CLASSIFIER_PROMPT


def classify_intent(query):
    # temperature 0 keeps the classifier stable between runs
    llm = get_llm(temperature=0)
    reply = llm.invoke(CLASSIFIER_PROMPT.format(query=query)).content.strip()
    if "GENERATE" in reply.upper():
        return "generate"
    return "explain"
