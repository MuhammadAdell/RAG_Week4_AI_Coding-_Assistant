import streamlit as st
from assistant import respond, extract_code
from code_runner import run_code
from knowledge import add_solution, get_collection
from memory import Memory

st.set_page_config(page_title="AI Coding Assistant", layout="wide")
st.title("Intelligent AI Coding Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = Memory()

with st.sidebar:
    st.subheader("Memory")
    mem = st.session_state.memory
    st.json(mem.preferences)
    st.write(f"Stored turns: {len(mem.turns)}, code snippets: {len(mem.codes)}")
    for question, _ in mem.turns:
        st.caption(question)
    st.divider()
    st.subheader("Knowledge base")
    st.write(f"{get_collection().count()} chunks in Chroma")
    if st.button("Reset chat"):
        st.session_state.messages = []
        st.session_state.memory = Memory()
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("intent"):
            badge = "Explain (no RAG)" if msg["intent"] == "explain" else "Generate (RAG)"
            st.caption(f"Route: {badge}")
            for s in msg.get("sources", []):
                st.caption(f"Retrieved from: {s['source']} chunk {s['chunk']}")

        if msg.get("code"):
            if st.button("Run code", key=f"run_{msg['id']}"):
                with st.spinner("Executing..."):
                    msg["output"], msg["stderr"] = run_code(msg["code"])
            if msg.get("output") is not None:
                st.markdown("**Output**")
                st.code(msg["output"] or "(no output)")
                if msg.get("stderr"):
                    st.markdown("**Stderr**")
                    st.code(msg["stderr"], language="text")

        if msg.get("needs_feedback"):
            with st.form(f"feedback_{msg['id']}"):
                st.markdown("Teach me the correct solution:")
                solution = st.text_area("Solution or reference", height=200)
                if st.form_submit_button("Add to knowledge base") and solution:
                    msg["learned"] = add_solution(solution)
                    msg["needs_feedback"] = False
                    st.rerun()
        if msg.get("learned"):
            st.success(f"Stored {msg['learned']} chunks. I will use them next time.")

query = st.chat_input("Ask me to explain code or generate code")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    memory = st.session_state.memory
    result = respond(query, memory)

    with st.chat_message("assistant"):
        answer = st.write_stream(result["stream"])

    msg = {
        "role": "assistant",
        "content": answer,
        "intent": result["intent"],
        "code": extract_code(answer),
        "sources": result["sources"],
        "needs_feedback": not result["relevant"],
        "id": len(st.session_state.messages),
    }
    st.session_state.messages.append(msg)

    memory.add_turn(query, answer)
    if msg["code"]:
        memory.add_code(msg["code"])
    memory.update(query, "python" if msg["code"] else "unknown", result["intent"])

    st.rerun()
# python3 -m streamlit run /Users/muhammadadel/Downloads/Muhammad_Adel_Cellula_Week4/app.py