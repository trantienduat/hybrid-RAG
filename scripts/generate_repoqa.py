import sys
import json
import random
import redis

def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_repoqa.py <repo_name> <output_file>")
        sys.exit(1)
        
    repo_name = sys.argv[1]
    output_file = sys.argv[2]
    
    r = redis.Redis(host='localhost', port=6379)
    # Query FalkorDB for functions with docstrings in the repo
    query = f"MATCH (f:Function) WHERE f.repository = '{repo_name}' AND f.docstring IS NOT NULL AND size(f.docstring) > 40 RETURN f.name, f.file_path, f.docstring"
    
    res = r.execute_command('GRAPH.QUERY', 'codebase', query)
    rows = res[1] if len(res) > 1 else []
    
    if not rows:
        print(f"No functions with docstrings found for {repo_name}!")
        sys.exit(1)
        
    print(f"Found {len(rows)} candidate functions for {repo_name}.")
    
    # Shuffle and select 50
    random.seed(42)
    selected = random.sample(rows, min(50, len(rows)))
    
    cases = []
    for idx, row in enumerate(selected, start=1):
        func_name = row[0].decode('utf-8')
        file_path = row[1].decode('utf-8')
        docstring = row[2].decode('utf-8')
        
        # Clean docstring (take the first sentence or first 120 chars)
        docstring = docstring.strip().split('\n')[0]
        if len(docstring) > 200:
            docstring = docstring[:197] + "..."
            
        cases.append({
            "id": f"repoqa-{repo_name}-{idx}",
            "question": docstring,
            "target_function": func_name,
            "file_path": file_path
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(cases)} cases and saved to {output_file}.")

if __name__ == '__main__':
    main()
