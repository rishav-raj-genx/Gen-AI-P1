#!/usr/bin/env python3
"""
Interactive CLI Chatbot for the RAG Document Knowledge Base
Ask questions and get answers grounded in the ingested document via vector search.
"""

import sys
from src.rag_engine import RAGEngine


def print_header():
    print("\n" + "=" * 75)
    print(" 🤖 RAG DOCUMENT CHATBOT (AI Agent Systems Knowledge Base)")
    print("=" * 75)
    print(" Ask any question based on the document, for example:")
    print("  • 'What is the ReAct pattern?'")
    print("  • 'What are the container sandboxing requirements?'")
    print("  • 'How does working memory handle garbage collection?'")
    print("  • 'What is the formula for cosine similarity?'")
    print(" Type 'exit' or 'quit' to leave, or '/stats' to see database stats.\n" + "-" * 75)


def run_interactive_chat():
    print("[*] Initializing RAG Engine and connecting to ChromaDB...")
    try:
        engine = RAGEngine(persist_dir="./vector_store")
    except Exception as e:
        print(f"[!] Error loading database: {e}")
        print("    Please run 'python pipeline.py' first to ingest the document.")
        sys.exit(1)

    print("[✓] RAG Engine ready! Vector database connected.\n")
    print_header()

    while True:
        try:
            user_input = input("\n💬 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("👋 Exiting chatbot. Have a great day!")
            break

        if user_input.lower() == "/stats":
            count = engine.db.count()
            print(f"\n📊 Database Stats:")
            print(f"   - Total Chunks Indexed: {count}")
            print(f"   - Storage Engine       : ChromaDB PersistentClient")
            print(f"   - Embedding Model      : {engine.embedder.MODEL_NAME}")
            print(f"   - Chunking Method      : Hierarchical Context-Aware Sliding Window")
            continue

        print("\n🔎 Searching vector database & synthesizing answer...")
        result = engine.answer_query(user_input, top_k=3)

        print("\n🤖 AI Assistant:")
        print("-" * 50)
        print(result["answer"])
        print("-" * 50)

        if result["sources"]:
            print(f"📚 Sources & Citations (Confidence: {result['confidence'] * 100:.1f}%):")
            for idx, src in enumerate(result["sources"][:2], 1):
                print(f"   [{idx}] {src['section']}")
                print(f"       Similarity: {src['similarity_score']:.4f} | Chunk ID: {src['chunk_id']}")


def main():
    if len(sys.argv) > 1:
        # Non-interactive CLI mode: e.g. python chatbot.py "What is ReAct?"
        query = " ".join(sys.argv[1:])
        engine = RAGEngine(persist_dir="./vector_store")
        result = engine.answer_query(query, top_k=2)
        print(f"\nQuestion: {query}\n")
        print(f"Answer:\n{result['answer']}\n")
        print(f"Top Source: {result.get('top_section')} (Score: {result.get('confidence'):.4f})\n")
    else:
        run_interactive_chat()


if __name__ == "__main__":
    main()
