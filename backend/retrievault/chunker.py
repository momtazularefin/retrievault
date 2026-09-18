"""AST-aware chunking of Python source into citation-exact spans.

Every chunk's ``code`` is exactly the source lines ``start_line..end_line`` of its file, so a
citation built from the span shows the text the model read. Spans never overlap, and every
non-blank line of a parseable file belongs to exactly one chunk, except comments trailing the
file's last statement.

Definitions are kept whole when they fit ``max_chars``. Larger definitions are split only on
syntax boundaries: a class into its own statements and its methods, a function into
signature parameters and body statements, and an oversized compound statement (a nested
function, ``if``, ``try``, loop) into the statements it contains. A simple statement is never
split, so one larger than the budget becomes a single oversized chunk.
"""

import ast
from dataclasses import dataclass

# Bump when chunk boundaries or chunk text change, so an index can be traced to its chunker.
CHUNKER_VERSION = "2"

# The embedding model and the reranker both truncate at 512 tokens. At the densest 5% of the
# FastAPI corpus (about 3.2 characters per token), 1,400 characters plus the context header
# stays inside that window.
DEFAULT_MAX_CHARS = 1400

MODULE_SYMBOL = "__module__"

DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def embedding_text(file_path: str, symbol_name: str, context: str, code: str) -> str:
    """Text used for embeddings and reranking: location header, context, then code."""
    header = f"{file_path} :: {symbol_name}"
    return "\n".join(piece for piece in (header, context, code) if piece)


@dataclass(frozen=True)
class Chunk:
    file_path: str
    symbol_name: str
    symbol_type: str  # module | class | function | method
    start_line: int
    end_line: int
    code: str
    # Enclosing signature lines (class header, def line) that are not part of the span but
    # tell a reader where the code sits.
    context: str = ""
    part: int = 1
    part_count: int = 1

    def embedding_text(self) -> str:
        return embedding_text(self.file_path, self.symbol_name, self.context, self.code)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "code": self.code,
            "context": self.context,
            "part": self.part,
            "part_count": self.part_count,
            "language": "python",
        }


class _FileChunker:
    def __init__(self, repo_relative_path: str, content: str, max_chars: int):
        self.path = repo_relative_path
        self.lines = content.splitlines()
        self.max_chars = max_chars
        self.chunks: list[Chunk] = []

    # -- span helpers ---------------------------------------------------------------------

    def text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start - 1 : end])

    def size(self, start: int, end: int) -> int:
        return len(self.text(start, end))

    def skip_blank(self, start: int, end: int) -> int:
        while start <= end and not self.lines[start - 1].strip():
            start += 1
        return start

    def emit(self, name, kind, start, end, context="", part=1, part_count=1):
        start = self.skip_blank(start, end)
        if start > end:
            return
        self.chunks.append(
            Chunk(
                file_path=self.path,
                symbol_name=name,
                symbol_type=kind,
                start_line=start,
                end_line=end,
                code=self.text(start, end),
                context=context,
                part=part,
                part_count=part_count,
            )
        )

    def pack(self, segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge adjacent (start, end) segments into groups that fit the size budget."""
        groups: list[tuple[int, int]] = []
        for start, end in segments:
            if groups and self.size(groups[-1][0], end) <= self.max_chars:
                groups[-1] = (groups[-1][0], end)
            else:
                groups.append((start, end))
        return groups

    def emit_groups(self, name, kind, segments, context=""):
        groups = [g for g in self.pack(segments) if self.skip_blank(*g) <= g[1]]
        for index, (start, end) in enumerate(groups, 1):
            self.emit(name, kind, start, end, context, index, len(groups))

    # -- structure ------------------------------------------------------------------------

    @staticmethod
    def node_start(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", [])
        return min([d.lineno for d in decorators] + [node.lineno])

    def body_segments(self, nodes: list[ast.stmt], first_line: int) -> list[tuple[int, int]]:
        """Contiguous spans for sibling statements; leading comments attach to the next node."""
        segments = []
        previous_end = first_line - 1
        for node in nodes:
            segments.append((previous_end + 1, node.end_lineno))
            previous_end = node.end_lineno
        return segments

    @staticmethod
    def child_statements(node: ast.AST) -> list[ast.stmt]:
        """Statements nested directly in a compound statement, including handler and case bodies."""
        children: list[ast.stmt] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                children.append(child)
            elif isinstance(child, (ast.excepthandler, ast.match_case)):
                children.extend(_FileChunker.child_statements(child))
        return sorted(children, key=lambda child: child.lineno)

    def expand(self, node: ast.stmt, start: int) -> list[tuple[int, int]]:
        """Segments for one statement, split at nested statements only when it is oversized.

        Lines between nested statements (``else:``, ``except ...:``) attach to the statement
        that follows them, so the segments stay contiguous.
        """
        end = node.end_lineno
        children = self.child_statements(node)
        if self.size(start, end) <= self.max_chars or not children:
            return [(start, end)]
        segments: list[tuple[int, int]] = []
        first_child = self.node_start(children[0])
        if first_child > start:
            segments.append((start, first_child - 1))
        previous_end = first_child - 1
        for child in children:
            child_start = max(previous_end + 1, start)
            segments.extend(self.expand(child, child_start))
            previous_end = child.end_lineno
        if previous_end < end:
            segments.append((previous_end + 1, end))
        return segments

    def expanded_segments(self, nodes: list[ast.stmt], first_line: int) -> list[tuple[int, int]]:
        segments = []
        for node, (start, _) in zip(nodes, self.body_segments(nodes, first_line), strict=True):
            segments.extend(self.expand(node, start))
        return segments

    def walk_block(self, nodes, first_line, owner, owner_kind, context):
        """Chunk a module or class body: definitions recurse, other statements are grouped.

        ``owner`` names the enclosing class (None at module level); runs of plain statements
        are emitted under the owner's name, or ``__module__`` at module level.
        """
        run: list[tuple[int, int]] = []
        run_name = owner or MODULE_SYMBOL
        run_kind = owner_kind
        segments = self.body_segments(nodes, first_line)
        for node, segment in zip(nodes, segments, strict=True):
            if isinstance(node, DEFINITION_TYPES):
                if run:
                    self.emit_groups(run_name, run_kind, run, context)
                    run = []
                start = self.skip_blank(segment[0], segment[1])
                self.definition(node, start, owner, context)
            else:
                run.extend(self.expand(node, segment[0]))
        if run:
            self.emit_groups(run_name, run_kind, run, context)

    def definition(self, node, start, owner, context):
        name = f"{owner}.{node.name}" if owner else node.name
        end = node.end_lineno
        if isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            kind = "method" if owner else "function"

        if self.size(start, end) <= self.max_chars:
            self.emit(name, kind, start, end, context)
            return

        body_start = self.node_start(node.body[0])
        if body_start <= node.lineno:
            # Body shares the definition line; there is no boundary to split on.
            self.emit(name, kind, start, end, context)
            return
        header_end = body_start - 1
        if isinstance(node, ast.ClassDef):
            header = self.text(self.node_start(node), header_end)
            self.split_class(node, name, start, header_end, header, context)
        else:
            self.split_function(node, name, kind, start, header_end, context)

    def split_class(self, node, name, start, header_end, header, context):
        inner_context = "\n".join(piece for piece in (context, header) if piece)
        members = node.body
        leading: list[ast.stmt] = []
        for member in members:
            if isinstance(member, DEFINITION_TYPES):
                break
            leading.append(member)

        # The class header travels with the leading statements (docstring, attributes), so
        # "what does this class inherit from" retrieves the chunk that holds the header.
        segments = self.body_segments(leading, header_end + 1)
        if segments:
            segments[0] = (start, segments[0][1])
        else:
            segments = [(start, header_end)]
        self.emit_groups(name, "class", segments, context)

        rest = members[len(leading) :]
        first_line = (segments[-1][1] + 1) if leading else header_end + 1
        self.walk_block(rest, first_line, name, "class", inner_context)

    def split_function(self, node, name, kind, start, header_end, context):
        def_line = self.lines[node.lineno - 1]
        inner_context = "\n".join(piece for piece in (context, def_line) if piece)

        signature = self.parameter_segments(node, start, header_end)
        body = self.expanded_segments(node.body, header_end + 1)
        groups = [g for g in self.pack(signature + body) if self.skip_blank(*g) <= g[1]]
        for index, (group_start, group_end) in enumerate(groups, 1):
            # Part one carries the def line itself; later parts name it in their context.
            part_context = context if index == 1 else inner_context
            self.emit(name, kind, group_start, group_end, part_context, index, len(groups))

    def parameter_segments(self, node, start, header_end) -> list[tuple[int, int]]:
        """Split a signature at parameter boundaries, keeping each default with its parameter."""
        args = node.args
        positional = args.posonlyargs + args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        params = list(zip(positional, defaults, strict=True))
        if args.vararg:
            params.append((args.vararg, None))
        params.extend(zip(args.kwonlyargs, args.kw_defaults, strict=True))
        if args.kwarg:
            params.append((args.kwarg, None))

        ends = []
        for arg, default in params:
            end = default.end_lineno if default is not None else arg.end_lineno
            ends.append(max(arg.end_lineno, end))
        ends = sorted({end for end in ends if end < header_end})

        segments = []
        previous_end = start - 1
        for end in ends:
            if end > previous_end:
                segments.append((previous_end + 1, end))
                previous_end = end
        segments.append((previous_end + 1, header_end))
        return segments


def chunk_file(
    file_path: str,
    repo_relative_path: str,
    content: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[Chunk]:
    """Chunk one Python file. Unparseable files yield no chunks."""
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return []
    chunker = _FileChunker(repo_relative_path, content, max_chars)
    chunker.walk_block(tree.body, 1, None, "module", "")
    return sorted(chunker.chunks, key=lambda chunk: chunk.start_line)
