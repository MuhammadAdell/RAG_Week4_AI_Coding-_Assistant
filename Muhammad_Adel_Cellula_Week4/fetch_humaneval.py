# Downloads the openai/openai_humaneval dataset (test split, 164 problems)
# from the Hugging Face dataset server and stores every problem directly in
# the Chroma database (chroma_db/chroma.sqlite3). No intermediate file:
# chunking and embeddings happen here, then the vectors go into sqlite.
# Run once before starting the app: python3 fetch_humaneval.py

import json
import urllib.request

from chromadb import PersistentClient

from build_kb import COLLECTION, DB_DIR, embeddings, splitter

BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "openai%2Fopenai_humaneval"
CONFIG = "openai_humaneval"


def fetch_rows():
    rows = []
    offset = 0
    while True:
        url = f"{BASE}?dataset={DATASET}&config={CONFIG}&split=test&offset={offset}&length=100"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = json.load(response)
        batch = [item["row"] for item in data["rows"]]
        rows.extend(batch)
        # the server returns 100 rows per call, stop when we have them all
        if not batch or len(rows) >= data["num_rows_total"]:
            break
        offset += len(batch)
    return rows


def problem_text(row):
    # solution = prompt + body, so each stored document is a complete example
    return (
        f"Problem {row['task_id']}, function {row['entry_point']}\n"
        "Prompt:\n" + row["prompt"].rstrip() + "\n"
        "Reference solution:\n" + row["prompt"].rstrip() + "\n"
        + row["canonical_solution"].rstrip()
    )


def store(rows):
    client = PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    total = 0
    for row in rows:
        chunks = splitter.split_text(problem_text(row))
        for i, chunk in enumerate(chunks):
            collection.upsert(
                ids=[f"{row['task_id'].replace('/', '_')}_chunk_{i}"],
                embeddings=embeddings.embed_documents([chunk]),
                documents=[chunk],
                metadatas=[{
                    "source": "openai_humaneval",
                    "problem": row["task_id"],
                    "entry_point": row["entry_point"],
                    "chunk": i,
                }],
            )
            total += 1
        print(f"{row['task_id']} ({row['entry_point']}): {len(chunks)} chunk(s)")
    print(f"Stored {total} chunks from {len(rows)} problems in {DB_DIR}/")


if __name__ == "__main__":
    store(fetch_rows())
