from typing import List, Dict, Any

def build_system_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Constructs the system prompt grounding the LLM in the retrieved chunks.
    Labels chunks as [S1], [S2], etc. and modifies the chunks to include 'citation_label'.
    """
    header = (
        "You are an expert coding assistant answering questions about the FastAPI codebase.\n"
        "You must answer the user's question strictly using ONLY the provided Source Chunks below.\n"
        "If the Source Chunks do not contain the answer, you must refuse to answer and state that "
        "the information is not available in the retrieved context.\n\n"
        "CRITICAL INSTRUCTIONS FOR CITATIONS:\n"
        "1. Every factual claim or reference to code MUST be cited using the exact label of the chunk (e.g., [S1], [S2]).\n"
        "2. Do NOT hallucinate citations. You may only use labels that appear below.\n"
        "3. Place the citation at the end of the relevant sentence.\n\n"
        "--- SOURCE CHUNKS ---\n\n"
    )
    
    body = ""
    for i, chunk in enumerate(chunks, 1):
        file_path = chunk.get("file_path", "unknown")
        start = chunk.get("start_line", "?")
        end = chunk.get("end_line", "?")
        symbol = chunk.get("symbol_name", "")
        code = chunk.get("code", "")
        
        label = f"[S{i}]"
        chunk["citation_label"] = label
        
        body += f"{label} {file_path} L{start}-L{end} ({symbol})\n"
        body += f"```python\n{code}\n```\n\n"
        
    return header + body
