import ast
from typing import List

class Chunk:
    def __init__(self, file_path: str, symbol_name: str, symbol_type: str, start_line: int, end_line: int, code: str):
        self.file_path = file_path
        self.symbol_name = symbol_name
        self.symbol_type = symbol_type
        self.start_line = start_line
        self.end_line = end_line
        self.code = code

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "language": "python"
        }

def get_node_source(node: ast.AST, source_lines: List[str]) -> str:
    start_lineno = node.lineno - 1
    end_lineno = node.end_lineno
    return "\n".join(source_lines[start_lineno:end_lineno])

def chunk_file(file_path: str, repo_relative_path: str, content: str) -> List[Chunk]:
    chunks = []
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return chunks

    source_lines = content.splitlines()
    
    # Extract preamble: imports, module docstring, assignments
    preamble_lines = []
    preamble_end_line = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            preamble_lines.append(get_node_source(node, source_lines))
            preamble_end_line = max(preamble_end_line, getattr(node, 'end_lineno', 0))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            # Module docstring
            preamble_lines.append(get_node_source(node, source_lines))
            preamble_end_line = max(preamble_end_line, getattr(node, 'end_lineno', 0))

    if preamble_lines:
        code = "\n".join(preamble_lines)
        chunks.append(Chunk(
            file_path=repo_relative_path,
            symbol_name="__module_preamble__",
            symbol_type="module",
            start_line=1,
            end_line=preamble_end_line,
            code=code
        ))

    # Extract functions and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            code = get_node_source(node, source_lines)
            chunks.append(Chunk(
                file_path=repo_relative_path,
                symbol_name=node.name,
                symbol_type="function",
                start_line=node.lineno,
                end_line=node.end_lineno,
                code=code
            ))
        elif isinstance(node, ast.ClassDef):
            # Check if class is "large" -> heuristic: > 100 lines or has many methods
            class_lines = node.end_lineno - node.lineno
            if class_lines > 100:
                # Split by method
                class_header = source_lines[node.lineno - 1]
                for subnode in node.body:
                    if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_code = get_node_source(subnode, source_lines)
                        # Prepend the real class header for context; the method source keeps
                        # its own indentation, so no manual re-indentation is needed.
                        code = f"{class_header}\n{method_code}"
                        chunks.append(Chunk(
                            file_path=repo_relative_path,
                            symbol_name=f"{node.name}.{subnode.name}",
                            symbol_type="method",
                            start_line=subnode.lineno,
                            end_line=subnode.end_lineno,
                            code=code
                        ))
            else:
                # Small class, keep whole
                code = get_node_source(node, source_lines)
                chunks.append(Chunk(
                    file_path=repo_relative_path,
                    symbol_name=node.name,
                    symbol_type="class",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    code=code
                ))

    return chunks
