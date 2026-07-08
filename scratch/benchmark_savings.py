import os
import sys
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_codebase_stats(root_dir: str):
    total_tokens = 0
    file_count = 0
    # Common text-based file extensions in codebase
    extensions = {'.py', '.html', '.css', '.js', '.json', '.sh', '.toml', '.md'}
    
    for root, dirs, files in os.walk(root_dir):
        # Exclude directories we don't want to scan (dependencies, cache, build)
        parts = root.split(os.sep)
        if any(part in parts for part in ['.venv', '.git', '__pycache__', '.pytest_cache', '.ruff_cache', 'fixtures']):
            continue
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in extensions:
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        tokens = count_tokens(content)
                        total_tokens += tokens
                        file_count += 1
                except Exception:
                    pass
    return file_count, total_tokens

if __name__ == "__main__":
    workspace = "/Volumes/Kioxia_SSD/SSD_workspace/Personal/hybrid-RAG"
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
        
    print(f"Analyzing codebase at: {workspace}")
    file_count, codebase_tokens = get_codebase_stats(workspace)
    
    # Average RAG query size: 5 retrieved chunks (~500 tokens each) + system prompt + query = ~3000 tokens
    rag_avg_tokens = 3000
    
    saved_tokens = max(0, codebase_tokens - rag_avg_tokens)
    saved_pct = (saved_tokens / codebase_tokens) * 100 if codebase_tokens > 0 else 0
    
    # Cost estimations: Average of Gemini 1.5 Pro and Claude 3.5 Sonnet (~$3 per 1 million input tokens)
    cost_per_million = 3.00
    naive_cost_1k = (codebase_tokens / 1_000_000) * cost_per_million * 1000
    rag_cost_1k = (rag_avg_tokens / 1_000_000) * cost_per_million * 1000
    saved_cost_1k = naive_cost_1k - rag_cost_1k
    
    print("-" * 65)
    print(f"Total Codebase Files Scanned: {file_count}")
    print(f"Total Codebase Size:          {codebase_tokens:,} tokens")
    print(f"Average RAG Prompt Size:      {rag_avg_tokens:,} tokens")
    print("-" * 65)
    print(f"Tokens Saved Per Query:       {saved_tokens:,} tokens")
    print(f"Token Saving Rate:            {saved_pct:.2f}%")
    print("-" * 65)
    print(f"Estimated API Input Cost (per 1,000 queries @ ${cost_per_million}/M tokens):")
    print(f"  - Without RAG (Naive Full Load):  ${naive_cost_1k:.2f}")
    print(f"  - With RAG (MCP Server):          ${rag_cost_1k:.2f}")
    print(f"  - Net Financial Saving:           ${saved_cost_1k:.2f}")
    print("-" * 65)
