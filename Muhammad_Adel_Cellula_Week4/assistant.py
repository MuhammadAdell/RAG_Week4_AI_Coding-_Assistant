from llm import get_llm
from prompts import EXPLAIN_PROMPT, GENERATE_PROMPT, NO_KNOWLEDGE_REPLY
from intent_classifier import classify_intent
from knowledge import retrieve, evaluate_relevance


def extract_code(text):
    parts = text.split("```")
    blocks = parts[1::2]  
    for block in blocks:
        block = block.strip("\n")
        if block.startswith("python"):
            return block.split("\n", 1)[1]
    for block in blocks:
        block = block.strip("\n")
        if "def " in block or "import " in block:
            return block
    return ""


def build_context(documents, sources):
    blocks = []
    for doc, meta in zip(documents, sources):
        blocks.append(f"[from {meta['source']}, chunk {meta['chunk']}]\n{doc}")
    return "\n\n---\n\n".join(blocks)


def respond(query, memory):
    intent = classify_intent(query)

    if intent == "explain":
        llm = get_llm()
        prompt = EXPLAIN_PROMPT.format(memory=memory.context(), query=query)
        return {
            "intent": "explain",
            "used_rag": False,
            "relevant": True,
            "sources": [],
            "stream": (chunk.content for chunk in llm.stream(prompt)),
        }

    documents, sources, _ = retrieve(query)
    context = build_context(documents, sources)
    if not evaluate_relevance(query, context):
        def single(text):
            yield text
        return {
            "intent": "generate",
            "used_rag": True,
            "relevant": False,
            "sources": sources,
            "stream": single(NO_KNOWLEDGE_REPLY),
        }

    llm = get_llm()
    prompt = GENERATE_PROMPT.format(
        context=context, memory=memory.context(), query=query
    )
    return {
        "intent": "generate",
        "used_rag": True,
        "relevant": True,
        "sources": sources,
        "stream": (chunk.content for chunk in llm.stream(prompt)),
    }
