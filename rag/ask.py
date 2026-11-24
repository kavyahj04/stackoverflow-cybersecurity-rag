"""Ask a question from the terminal.

Usage (from rag-dev):  python -m rag.ask "How do I prevent SQL injection in PHP?"
"""
import argparse
import time

from rag.db import connect
from rag.generation.answer import RagPipeline
from rag.retrieval.bm25 import BM25
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.store import ThreadStore


def build_pipeline(conn) -> RagPipeline:
    start = time.time()
    retriever = HybridRetriever(DenseRetriever(conn), BM25.from_database(conn))
    print(f"(search index ready in {time.time() - start:.0f}s)\n", flush=True)
    return RagPipeline(retriever, ThreadStore(conn))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    args = parser.parse_args()

    answer = build_pipeline(connect()).ask(args.question)
    print(answer.render())
    r, u, c = answer.retrieval, answer.usage, answer.citations
    print(f"\n--- {u.get('seconds')}s | model {u.get('model', '-')} | "
          f"tokens in/out {u.get('prompt_tokens', 0)}/{u.get('completion_tokens', 0)} | "
          f"best distance {r['best_distance']} | exact match {r['exact_match']} | "
          f"threads in context {r.get('threads_in_context', 0)} | "
          f"invalid citations removed {c.invalid} | uncited sentences {c.uncited_sentences}")


if __name__ == "__main__":
    main()
