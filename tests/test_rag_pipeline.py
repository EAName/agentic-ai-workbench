"""Tests for the RAG pipeline."""

import tempfile
from pathlib import Path

import pytest
from memory.rag.pipeline import RAGPipeline, Document


class TestChunking:
    def test_basic_chunking(self):
        rag = RAGPipeline(chunk_size=100, chunk_overlap=20)
        text = "A" * 250
        chunks = rag.chunk_text(text, doc_id="test")

        assert len(chunks) >= 2
        assert all(isinstance(c, Document) for c in chunks)
        assert chunks[0].doc_id == "test"
        assert chunks[0].chunk_index == 0

    def test_empty_text(self):
        rag = RAGPipeline()
        chunks = rag.chunk_text("", doc_id="empty")
        assert len(chunks) == 0

    def test_short_text(self):
        rag = RAGPipeline(chunk_size=500)
        chunks = rag.chunk_text("Short text.", doc_id="short")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."


class TestIngestion:
    def test_ingest_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("This is a test document for RAG ingestion.")

            rag = RAGPipeline(
                persist_dir=str(Path(tmpdir) / "embeddings"),
                chunk_size=100,
            )
            count = rag.ingest_file(test_file)
            assert count >= 1

    def test_ingest_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "docs").mkdir()
            (Path(tmpdir) / "docs" / "file1.txt").write_text("Document one content.")
            (Path(tmpdir) / "docs" / "file2.md").write_text("# Document Two\n\nMore content here.")

            rag = RAGPipeline(
                persist_dir=str(Path(tmpdir) / "embeddings"),
                chunk_size=100,
            )
            total = rag.ingest_directory(Path(tmpdir) / "docs")
            assert total >= 2


class TestRetrieval:
    def test_retrieve_after_ingest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text(
                "Federated learning is a machine learning approach "
                "where a model is trained across multiple decentralized "
                "devices or servers holding local data samples."
            )

            rag = RAGPipeline(
                persist_dir=str(Path(tmpdir) / "embeddings"),
                chunk_size=200,
            )
            rag.ingest_file(test_file)

            results = rag.retrieve("What is federated learning?", top_k=3)
            assert len(results) >= 1
            assert results[0].score > 0

    def test_format_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Important fact about AI agents and memory systems.")

            rag = RAGPipeline(
                persist_dir=str(Path(tmpdir) / "embeddings"),
                chunk_size=200,
            )
            rag.ingest_file(test_file)

            results = rag.retrieve("AI agents", top_k=3)
            context = rag.format_context(results)

            assert "Retrieved Context" in context
            assert len(context) > 0
