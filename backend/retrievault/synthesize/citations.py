import re
from typing import List, Dict, Any, Tuple
from retrievault.config import get_settings

def extract_and_validate_citations(answer: str, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extracts [S#] citations from the answer, validates them against the provided chunks.
    Returns (valid_citations, error_message). 
    If error_message is empty, citations are valid.
    """
    # Find all unique [S#] markers
    matches = set(re.findall(r'\[S\d+\]', answer))
    
    chunk_map = {c["citation_label"]: c for c in chunks if "citation_label" in c}
    
    settings = get_settings()
    repo = settings.corpus_repo
    tag = settings.corpus_tag
    
    valid_citations = []
    hallucinated = []
    
    for match in matches:
        if match in chunk_map:
            c = chunk_map[match]
            file_path = c.get("file_path", "")
            start_line = c.get("start_line", 1)
            end_line = c.get("end_line", 1)
            # construct github url: https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/routing.py#L40-L50
            github_url = f"https://github.com/{repo}/blob/{tag}/{file_path}#L{start_line}-L{end_line}"
            valid_citations.append({
                "label": match,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "symbol_name": c.get("symbol_name", ""),
                "github_url": github_url
            })
        else:
            hallucinated.append(match)
            
    if hallucinated:
        return [], f"You cited {', '.join(sorted(hallucinated))} which do not exist in the source chunks. Please revise your answer to ONLY use the provided [S#] labels."
        
    return valid_citations, ""
