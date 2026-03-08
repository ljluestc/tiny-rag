"""Interview RAG — Build and query a RAG database from interview Q&A.

Covers: GenAI, RAG, LLM, distributed training, ML, system design, DevOps.

Usage:
    # Build the interview vector database
    python script/interview_rag.py -t build -c config/interview_config.json -p data/interview_questions/interview_qa.json

    # Search interactively
    python script/interview_rag.py -t search -c config/interview_config.json

    # Single query
    python script/interview_rag.py -t query -c config/interview_config.json -q "What is RAG?"
"""

import sys
sys.path.append(".")

import os
import json
import argparse
from loguru import logger

from tinyrag import RAGConfig, TinyRAG
from tinyrag.utils import read_json_to_list


INTERVIEW_DATA_PATH = "data/interview_questions/interview_qa.json"


def load_interview_data(data_path: str) -> list:
    """Load interview Q&A data and format each record as a searchable text block.

    Supports two JSON formats:
    1. Simple: [{"completion": "Q: ...\nA: ..."}, ...]
    2. Structured: [{"section": "...", "question": "...", "completion": "..."}, ...]
    """
    raw_data = read_json_to_list(data_path)
    text_list = []
    for item in raw_data:
        if "question" in item:
            # Structured format with section/question/completion fields
            section = item.get("section", "General")
            question = item.get("question", "")
            answer = item.get("completion", "")
            text_block = (
                f"[{section}] Q: {question}\n"
                f"A: {answer}"
            )
            text_list.append(text_block)
        elif "completion" in item:
            # Simple format: completion contains the full Q&A text
            text_list.append(item["completion"])
        else:
            logger.warning(f"Skipping unrecognized record format: {list(item.keys())}")
    logger.info(f"Loaded {len(text_list)} interview Q&A records from {data_path}")
    return text_list


def build_db(config_path: str, data_path: str):
    """Build the interview RAG vector database."""
    text_list = load_interview_data(data_path)

    config = read_json_to_list(config_path)
    rag_config = RAGConfig(**config)

    # For interview data, skip sentence splitting — each Q&A is already a
    # self-contained chunk.  We directly build the searcher DB.
    from tinyrag import Searcher
    searcher = Searcher(
        emb_model_id=rag_config.emb_model_id,
        ranker_model_id=rag_config.ranker_model_id,
        device=rag_config.device,
        base_dir=rag_config.base_dir,
    )
    logger.info("Building interview database...")
    searcher.build_db(text_list)
    logger.info("Saving interview database...")
    searcher.save_db()
    logger.info("Interview database built and saved successfully!")


def search_interactive(config_path: str):
    """Interactive search loop against the interview RAG database."""
    config = read_json_to_list(config_path)
    rag_config = RAGConfig(**config)

    from tinyrag import Searcher
    searcher = Searcher(
        emb_model_id=rag_config.emb_model_id,
        ranker_model_id=rag_config.ranker_model_id,
        device=rag_config.device,
        base_dir=rag_config.base_dir,
    )
    searcher.load_db()
    logger.info("Interview database loaded. Type 'quit' to exit.\n")

    while True:
        query = input("Ask a DevOps interview question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        results = searcher.search(query, top_n=3)
        print("\n" + "=" * 80)
        for rank, (score, text) in enumerate(results, 1):
            print(f"\n--- Result #{rank}  (score: {score:.4f}) ---")
            print(text)
        print("=" * 80 + "\n")


def single_query(config_path: str, query: str):
    """Run a single query and print top results."""
    config = read_json_to_list(config_path)
    rag_config = RAGConfig(**config)

    from tinyrag import Searcher
    searcher = Searcher(
        emb_model_id=rag_config.emb_model_id,
        ranker_model_id=rag_config.ranker_model_id,
        device=rag_config.device,
        base_dir=rag_config.base_dir,
    )
    searcher.load_db()

    results = searcher.search(query, top_n=3)
    print(f"\nQuery: {query}\n")
    for rank, (score, text) in enumerate(results, 1):
        print(f"--- Result #{rank}  (score: {score:.4f}) ---")
        print(text)
        print()


def main():
    parser = argparse.ArgumentParser(description="Interview RAG — Technical Q&A retrieval")
    parser.add_argument("-c", "--config", type=str, default="config/interview_config.json",
                        help="RAG config file path")
    parser.add_argument("-t", "--type", type=str, default="search",
                        choices=["build", "search", "query"],
                        help="Operation type: build, search (interactive), query (single)")
    parser.add_argument("-p", "--path", type=str, default=INTERVIEW_DATA_PATH,
                        help="Interview data JSON path (for build)")
    parser.add_argument("-q", "--query", type=str, default="",
                        help="Query string (for single query mode)")

    args = parser.parse_args()

    if args.type == "build":
        build_db(args.config, args.path)
    elif args.type == "search":
        search_interactive(args.config)
    elif args.type == "query":
        if not args.query:
            print("ERROR: --query is required for query mode")
            sys.exit(1)
        single_query(args.config, args.query)


if __name__ == "__main__":
    main()
