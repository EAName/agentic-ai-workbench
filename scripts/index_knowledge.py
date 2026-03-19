"""
Knowledge Base Indexer.

Index documents from data/knowledge_base/ into the vector store
so agents can access them via RAG.

Usage:
    python scripts/index_knowledge.py
    python scripts/index_knowledge.py --dir ./custom/docs --collection my_project
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from memory.rag.pipeline import RAGPipeline
from config.settings import settings


def resolve_doc_dir(path_arg: str) -> Path:
    """Resolve docs directory from cwd or project root."""
    path = Path(path_arg).expanduser()
    if path.exists():
        return path
    root_candidate = PROJECT_ROOT / path_arg
    if root_candidate.exists():
        return root_candidate
    return path


def main():
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument(
        "--dir",
        type=str,
        default="data/knowledge_base",
        help="Directory of documents to index",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="knowledge_base",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks",
    )
    args = parser.parse_args()

    rag = RAGPipeline(
        persist_dir=settings.vector_store.chroma_persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    doc_dir = resolve_doc_dir(args.dir)
    if not doc_dir.exists():
        print(f"Directory not found: {args.dir}")
        print("Create it and add your documents (.txt, .md, .pdf, .docx, .py, .yaml, .json)")
        sys.exit(1)

    print(f"Indexing documents from: {args.dir}")
    print(f"Collection: {args.collection}")
    print(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    print()

    total = rag.ingest_directory(doc_dir)
    print(f"\nDone. Indexed {total} chunks into '{args.collection}'.")
    print(f"Embeddings stored at: {settings.vector_store.chroma_persist_dir}")


if __name__ == "__main__":
    main()
