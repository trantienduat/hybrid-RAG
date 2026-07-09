#!/usr/bin/env python3
import argparse
import io
import os
import sys
import zipfile
import httpx
import tiktoken

# Common directories to skip during local scanning
EXCLUDE_DIRS = {
    ".git", ".github", ".venv", ".pytest_cache", ".ruff_cache", 
    "__pycache__", "node_modules", "dist", "build", "fixtures"
}

def count_tokens(text: str, model: str) -> int:
    """Accurately count tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def scan_local_directory(path: str, extensions: set, model: str) -> tuple[int, int]:
    """Scan local file system for files matching extensions."""
    total_tokens = 0
    file_count = 0
    
    for root, dirs, files in os.walk(path):
        # In-place modification to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        total_tokens += count_tokens(content, model)
                        file_count += 1
                except Exception:
                    pass
    return file_count, total_tokens

def scan_github_zip(zip_url: str, extensions: set, model: str) -> tuple[int, int]:
    """Download and scan files from a remote GitHub ZIP in memory."""
    print(f"Downloading ZIP from remote URL: {zip_url} ...")
    headers = {"User-Agent": "hybrid-rag-benchmarker"}
    resp = httpx.get(zip_url, follow_redirects=True, headers=headers, timeout=60.0)
    resp.raise_for_status()
    
    print("Extracting and analyzing codebase...")
    total_tokens = 0
    file_count = 0
    
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for file_info in z.infolist():
            if file_info.is_dir():
                continue
            
            parts = file_info.filename.split("/")
            if any(p in EXCLUDE_DIRS for p in parts):
                continue
                
            ext = os.path.splitext(file_info.filename)[1].lower()
            if ext in extensions:
                try:
                    with z.open(file_info) as f:
                        content = f.read().decode("utf-8", errors="ignore")
                        total_tokens += count_tokens(content, model)
                        file_count += 1
                except Exception:
                    pass
                    
    return file_count, total_tokens

def main():
    parser = argparse.ArgumentParser(
        description="Standardized Token Savings & API Cost Benchmarker for Hybrid RAG."
    )
    parser.add_argument(
        "target",
        help="Local directory path, 'owner/repo' GitHub string, or a direct ZIP URL."
    )
    parser.add_argument(
        "--extensions",
        default=".py,.java,.ts,.js,.json,.toml,.md",
        help="Comma-separated file extensions to scan (default: .py,.java,.ts,.js,.json,.toml,.md)."
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Tiktoken model encoding to count tokens with (default: gpt-4)."
    )
    parser.add_argument(
        "--cost",
        type=float,
        default=3.00,
        help="LLM API input cost per 1 million tokens in USD (default: 3.00)."
    )
    parser.add_argument(
        "--rag-size",
        type=int,
        default=3000,
        help="Estimated average RAG prompt size in tokens (default: 3000)."
    )
    args = parser.parse_args()

    extensions = {ext.strip().lower() for ext in args.extensions.split(",")}
    target = args.target
    
    # Resolve target
    if os.path.isdir(target):
        print(f"Scanning local directory: {os.path.abspath(target)}")
        file_count, total_tokens = scan_local_directory(target, extensions, args.model)
    else:
        # Check if owner/repo format
        if "/" in target and not target.startswith("http"):
            zip_url = f"https://github.com/{target}/archive/refs/heads/master.zip"
            # Fallback to main branch check if needed
            print(f"Resolving GitHub shorthand '{target}' to: {zip_url}")
        else:
            zip_url = target
            
        try:
            file_count, total_tokens = scan_github_zip(zip_url, extensions, args.model)
        except Exception as exc:
            # Try falling back to 'main' branch if 'master' failed
            if "master.zip" in zip_url:
                alt_url = zip_url.replace("master.zip", "main.zip")
                print(f"Retrying with main branch: {alt_url} ...")
                try:
                    file_count, total_tokens = scan_github_zip(alt_url, extensions, args.model)
                except Exception:
                    print(f"Error: Failed to process remote target: {exc}", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"Error: Failed to process remote target: {exc}", file=sys.stderr)
                sys.exit(1)

    # Computations
    saved_tokens = max(0, total_tokens - args.rag_size)
    saved_pct = (saved_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    naive_cost_1k = (total_tokens / 1_000_000) * args.cost * 1000
    rag_cost_1k = (args.rag_size / 1_000_000) * args.cost * 1000
    saved_cost_1k = naive_cost_1k - rag_cost_1k

    # Print Results Table
    print("\n" + "=" * 65)
    print(f" HYBRID RAG TOKEN SAVINGS BENCHMARK: {target.upper()}")
    print("=" * 65)
    print(f" Scanned Files Count:         {file_count:,}")
    print(f" Total Codebase Footprint:    {total_tokens:,} tokens")
    print(f" Average RAG Query Prompt:    {args.rag_size:,} tokens")
    print("-" * 65)
    print(f" Net Tokens Saved Per Query:  {saved_tokens:,} tokens")
    print(f" Token Compression Rate:      {saved_pct:.2f}%")
    print("-" * 65)
    print(f" Estimated API Input Cost (per 1,000 queries @ ${args.cost:.2f}/M tokens):")
    print(f"   - Naive Loading (No RAG):  ${naive_cost_1k:.2f}")
    print(f"   - Hybrid RAG (MCP Scoped): ${rag_cost_1k:.2f}")
    print(f"   - Net Financial Savings:   ${saved_cost_1k:.2f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
