"""
apps/ingest/ingest.py
One-shot data ingestion pipeline. Run this before starting the voice or chat app.

Sources ingested:
  1. Resume PDF (PyMuPDF — section-aware chunking)
  2. GitHub repos (READMEs + commit messages via PyGitHub)
  3. Upserts all chunks to Pinecone via OpenAI embeddings

Usage:
    python ingest.py --resume data/resume.pdf --github-user your-username

Environment variables required:
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX, GITHUB_TOKEN, GITHUB_USERNAME
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import List, Tuple

from pypdf import PdfReader
from github import Github
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Shared embedding helper
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))
from rag.retrieve import embed_texts, PINECONE_DIM

load_dotenv()


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 100   # increased from 50 to avoid boundary splits (Failure 1 fix)
BATCH_SIZE    = 100   # Pinecone upsert batch size

# Section-aware markers for resume chunking
SECTION_MARKERS = [
    "Education", "Experience", "Work Experience", "Projects",
    "Skills", "Certifications", "Awards", "Publications"
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ──────────────────────────────────────────────
# Source 1: Resume PDF
# ──────────────────────────────────────────────
def ingest_resume(pdf_path: str) -> List[Tuple[str, dict]]:
    """Extract and chunk resume PDF. Returns (text, metadata) pairs."""
    print(f"[INGEST] Reading resume: {pdf_path}")
    reader = PdfReader(pdf_path)

    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    # Detect current section for metadata
    lines = full_text.split("\n")
    current_section = "General"
    section_texts: dict[str, list[str]] = {"General": []}

    for line in lines:
        stripped = line.strip()
        if any(stripped.lower().startswith(m.lower()) for m in SECTION_MARKERS):
            current_section = stripped
            section_texts[current_section] = []
        else:
            section_texts.setdefault(current_section, []).append(line)

    docs: List[Tuple[str, dict]] = []
    for section, text_lines in section_texts.items():
        section_text = "\n".join(text_lines).strip()
        if not section_text:
            continue
        chunks = splitter.split_text(section_text)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                docs.append((chunk, {
                    "source":  "resume",
                    "section": section,
                    "chunk":   i,
                }))

    print(f"[INGEST] Resume: {len(docs)} chunks from {len(section_texts)} sections")
    return docs


def ingest_resume_txt(txt_path: str) -> List[Tuple[str, dict]]:
    """Extract and chunk resume text. Returns (text, metadata) pairs."""
    print(f"[INGEST] Reading resume text: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Detect current section for metadata
    lines = full_text.split("\n")
    current_section = "General"
    section_texts: dict[str, list[str]] = {"General": []}

    for line in lines:
        stripped = line.strip()
        if any(stripped.lower().startswith(m.lower()) for m in SECTION_MARKERS):
            current_section = stripped
            section_texts[current_section] = []
        else:
            section_texts.setdefault(current_section, []).append(line)

    docs: List[Tuple[str, dict]] = []
    for section, text_lines in section_texts.items():
        section_text = "\n".join(text_lines).strip()
        if not section_text:
            continue
        chunks = splitter.split_text(section_text)
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                docs.append((chunk, {
                    "source":  "resume",
                    "section": section,
                    "chunk":   i,
                }))

    print(f"[INGEST] Resume text: {len(docs)} chunks from {len(section_texts)} sections")
    return docs


# ──────────────────────────────────────────────
# Source 2: GitHub repos
# ──────────────────────────────────────────────
def ingest_github(username: str, token: str, max_repos: int = 20) -> List[Tuple[str, dict]]:
    """Fetch READMEs, package.json/requirements.txt, and recent commit messages."""
    print(f"[INGEST] Fetching GitHub repos for {username}...")
    gh = Github(token)
    user = gh.get_user(username)
    repos = list(user.get_repos())[:max_repos]

    docs: List[Tuple[str, dict]] = []

    for repo in repos:
        if repo.fork:
            continue  # skip forks — only index original work

        meta_base = {
            "source":      "github",
            "repo":        repo.full_name,
            "stars":       repo.stargazers_count,
            "language":    repo.language or "unknown",
            "last_commit": str(repo.pushed_at),
        }

        # README
        readme_content = ""
        try:
            readme_content = repo.get_readme().decoded_content.decode("utf-8", errors="ignore")
            for i, chunk in enumerate(splitter.split_text(readme_content)):
                if chunk.strip():
                    docs.append((chunk, {**meta_base, "file": "README.md", "chunk": i}))
        except Exception:
            pass

        # Repository Summary (describes the project as a whole)
        repo_desc = repo.description
        if not repo_desc and readme_content:
            import re
            # Extract first few sentences or lines of README as description
            clean_text = re.sub(r'```.*?```', '', readme_content, flags=re.DOTALL)
            clean_text = re.sub(r'<.*?>', '', clean_text)
            lines = [l.strip() for l in clean_text.split('\n')]
            desc_lines = []
            for l in lines:
                if not l:
                    continue
                if l.startswith(('#', '*', '-', '>')):
                    continue
                desc_lines.append(l)
                if len(desc_lines) >= 3:
                    break
            if desc_lines:
                repo_desc = " ".join(desc_lines)
                if len(repo_desc) > 200:
                    repo_desc = repo_desc[:200] + "..."

        repo_desc = repo_desc or "No description provided."
        repo_summary = f"GitHub repository: {repo.name} ({repo.full_name}). Description: {repo_desc} Primary Language: {repo.language or 'unknown'}. Stars: {repo.stargazers_count}. Forks: {repo.forks_count}."
        docs.append((repo_summary, {**meta_base, "file": "summary"}))

        # package.json or requirements.txt (tech stack signals)
        for dep_file in ["package.json", "requirements.txt", "pyproject.toml"]:
            try:
                content = repo.get_contents(dep_file).decoded_content.decode("utf-8", errors="ignore")
                # Just take first 512 chars — enough for tech stack info
                snippet = content[:512]
                docs.append((f"[{dep_file} for {repo.full_name}]\n{snippet}", {
                    **meta_base, "file": dep_file
                }))
            except Exception:
                pass

        # Last 15 commit messages (coding patterns & history, ignoring trivial ones)
        try:
            for commit in repo.get_commits()[:15]:
                msg = commit.commit.message.strip()
                if len(msg) > 35:  # skip trivial messages like "Update README.md" or "first commit"
                    docs.append((
                        f"Git commit in {repo.full_name}: {msg}",
                        {**meta_base, "file": "commits", "sha": commit.sha[:8]}
                    ))
        except Exception:
            pass

    print(f"[INGEST] GitHub: {len(docs)} chunks from {len(repos)} repos")
    return docs


# ──────────────────────────────────────────────
# Upsert to Pinecone
# ──────────────────────────────────────────────
def upsert_to_pinecone(docs: List[Tuple[str, dict]]) -> None:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX", "persona")
    region = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

    # Create index if it doesn't exist
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"[INGEST] Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )

    index = pc.Index(index_name)

    print(f"[INGEST] Embedding and upserting {len(docs)} chunks...")
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        batch = docs[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
        texts = [d[0] for d in batch]

        t0 = time.time()
        embeddings = embed_texts(texts)
        elapsed = (time.time() - t0) * 1000

        vectors = []
        for i, (text, meta) in enumerate(batch):
            doc_id = hashlib.md5(text.encode()).hexdigest()
            vectors.append({
                "id":     doc_id,
                "values": embeddings[i],
                "metadata": {"text": text, **meta},
            })

        index.upsert(vectors=vectors)
        print(f"[INGEST] Batch {batch_num + 1}/{total_batches} upserted | embed: {elapsed:.0f}ms")

    stats = index.describe_index_stats()
    print(f"\n[INGEST] Done. Total vectors in index: {stats['total_vector_count']}")


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest resume + GitHub into Pinecone")
    parser.add_argument("--resume",       default="data/resume.pdf",  help="Path to resume PDF")
    parser.add_argument("--github-user",  default=os.environ.get("GITHUB_USERNAME", ""), help="GitHub username")
    parser.add_argument("--github-token", default=os.environ.get("GITHUB_TOKEN", ""),    help="GitHub PAT")
    parser.add_argument("--max-repos",    type=int, default=20, help="Max repos to index")
    parser.add_argument("--skip-github",  action="store_true", help="Only ingest resume")
    args = parser.parse_args()

    all_docs: List[Tuple[str, dict]] = []

    # Resume
    if os.path.exists(args.resume):
        if args.resume.endswith(".pdf"):
            all_docs.extend(ingest_resume(args.resume))
        else:
            all_docs.extend(ingest_resume_txt(args.resume))
    else:
        txt_fallback = args.resume.replace(".pdf", ".txt")
        if args.resume == "data/resume.pdf" and os.path.exists(txt_fallback):
            all_docs.extend(ingest_resume_txt(txt_fallback))
        else:
            print(f"[WARN] Resume not found at {args.resume} (or fallback {txt_fallback}), skipping.")

    # GitHub
    if not args.skip_github and args.github_user and args.github_token:
        all_docs.extend(ingest_github(args.github_user, args.github_token, args.max_repos))
    elif not args.skip_github:
        print("[WARN] GITHUB_USERNAME or GITHUB_TOKEN not set, skipping GitHub ingestion.")

    if not all_docs:
        print("[ERROR] No documents to ingest. Exiting.")
        sys.exit(1)

    upsert_to_pinecone(all_docs)
    print(f"\n✅ Ingested {len(all_docs)} total chunks.")
