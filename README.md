# Custom RAG Ingestion Pipeline & Chunking Engine

A production-grade, modular RAG (Retrieval-Augmented Generation) ingestion pipeline that processes an original technical document through a 7-stage architecture:

$$\text{Upload} \longrightarrow \text{Load} \longrightarrow \text{Split} \longrightarrow \text{Clean} \longrightarrow \text{Chunk} \longrightarrow \text{Embedding} \longrightarrow \text{Database}$$

---

## 1. Chunking Method Used

### Name of the Method
> **Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping**  
> *(Abbreviated: HC-SW Chunking)*

### Why This Method?
Standard chunking approaches in RAG suffer from three fundamental flaws:
1. **Naive Fixed-Size Character / Word Chunking:** Blindly chops sentences, entities, and code blocks in half at arbitrary indices, destroying semantic coherence.
2. **Naive Recursive Character Chunking:** Slices text without maintaining the hierarchical parentage (headings, section titles). Chunks lose their contextual grounding (e.g., a chunk talking about "Timeout Error 504" forgets whether it belongs to Database Replication or User Authentication).
3. **Naive Paragraph Chunking:** Results in unpredictable chunk dimensions—some chunks are tiny (e.g., 20 characters), while others exceed the token limit of the embedding model.

### How Our Custom Chunking Method Works:
1. **Structural Scope Preservation:** Blocks are grouped by their document breadcrumb ancestry (`H1 > H2 > H3`). Chunks never bleed across major unrelated section boundaries.
2. **Atomic Semantic Unit Decomposition:** Text is decomposed into atomic semantic units:
   - Paragraphs are split into complete sentences using regex lookbehinds `(?<=[.!?])\s+`.
   - Lists are split into individual items (`- ...`, `1. ...`).
   - Code blocks are preserved intact as atomic blocks.
3. **Dynamic Budget Accumulation with Sentence Snapping:** Atomic units are accumulated greedily up to `max_chunk_size` (e.g. 450 characters). When adding another sentence would breach the character budget, the chunk **snaps cleanly at the last complete sentence boundary**. No sentence is ever split mid-thought.
4. **Context-Aware Sliding Window Overlap:** Instead of a fixed arbitrary character overlap, the sliding window inspects the trailing sentences of chunk $k$ and selects complete sentences matching the target `overlap_size` (e.g. 110 characters) as the prefix of chunk $k+1$.
5. **Contextual Breadcrumb Injection:** Each chunk generates a dual representation:
   - `content`: Raw clean chunk text.
   - `contextual_text`: Prefixed with the document hierarchy:  
     `[Context: Autonomous AI Agent Systems > 1. Cognitive Architecture > 1.1 The Extended ReAct Pattern]`  
     This ensures the embedding model encodes both domain ancestry and local details.

---

## 2. Pipeline Stages

| Stage | Module | Responsibility |
|---|---|---|
| **1. Upload** | `src/uploader.py` | Validates file existence, MIME type, size limit, generates SHA-256 hash and stages file to `./uploads/`. |
| **2. Load** | `src/loader.py` | Detects encoding (`utf-8`), decodes raw bytes, extracts document-level metrics (lines, words, characters). |
| **3. Split** | `src/splitter.py` | Decomposes document by markdown headers (`#`, `##`, `###`), tracking active breadcrumbs and block types (`header`, `paragraph`, `code`, `list`). |
| **4. Clean** | `src/cleaner.py` | Normalizes whitespace, strips horizontal rules, normalizes smart quotes/dashes, eliminates zero-width spaces, and preserves code syntax. |
| **5. Chunk** | `src/chunker.py` | Executes **Hierarchical Context-Aware Sliding Window Chunking with Semantic Boundary Snapping**. |
| **6. Embedding** | `src/embedder.py` | Encodes contextual chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2`. |
| **7. Database** | `src/database.py` | Persists vectors, text, and hierarchical metadata into a persistent ChromaDB vector store; executes cosine similarity queries. |

---

## 3. How to Use the Chatbot

You have two options to query and chat with the document knowledge base:

### Option A: Interactive Terminal Chatbot
Run the interactive CLI chatbot directly in your terminal:
```bash
.venv/bin/python chatbot.py
```
Or ask a one-off question directly:
```bash
.venv/bin/python chatbot.py "What is the ReAct loop?"
```

### Option B: Modern Web UI Chatbot (Browser)
Launch the local web server:
```bash
.venv/bin/python web_chatbot.py
```
Then open your browser and navigate to:
**http://localhost:8080**

Features of the Web Chatbot:
* Real-time semantic search against ChromaDB.
* Grounded answers with exact citations and confidence scores.
* Expandable sources showing the section breadcrumb, chunk ID, and snippet.
* Quick preset question pills for instant testing.

---

## 4. Running the Pipeline & Tests

### Re-run Ingestion Pipeline:
```bash
.venv/bin/python pipeline.py
```

### Run Automated Unit Tests:
```bash
.venv/bin/python -m unittest test_pipeline.py
```
# Gen-AI-P1
