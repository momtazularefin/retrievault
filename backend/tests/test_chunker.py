import textwrap

from retrievault.chunker import MODULE_SYMBOL, chunk_file


def chunks_for(source: str, max_chars: int = 1400):
    content = textwrap.dedent(source).lstrip("\n")
    return content, chunk_file("test.py", "fastapi/test.py", content, max_chars=max_chars)


def assert_span_invariants(content: str, chunks) -> None:
    """Every chunk's code is its exact span; spans never overlap; non-blank lines are covered."""
    lines = content.splitlines()
    owner: dict[int, str] = {}
    for chunk in chunks:
        assert chunk.code == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        for line_number in range(chunk.start_line, chunk.end_line + 1):
            assert line_number not in owner, f"line {line_number} is in two chunks"
            owner[line_number] = chunk.symbol_name
    for line_number, line in enumerate(lines, 1):
        if line.strip():
            assert line_number in owner, f"line {line_number} is in no chunk: {line!r}"


def test_small_definitions_stay_whole_with_module_statements():
    content, chunks = chunks_for(
        """
        import os

        def my_func():
            return 1

        class MyClass:
            def method(self):
                pass
        """
    )

    assert [(c.symbol_name, c.symbol_type) for c in chunks] == [
        (MODULE_SYMBOL, "module"),
        ("my_func", "function"),
        ("MyClass", "class"),
    ]
    assert (chunks[1].start_line, chunks[1].end_line) == (3, 4)
    assert (chunks[2].start_line, chunks[2].end_line) == (6, 8)
    assert_span_invariants(content, chunks)


def test_decorators_and_leading_comments_belong_to_the_definition():
    content, chunks = chunks_for(
        """
        import functools

        # Cache the lookup.
        @functools.cache
        def lookup():
            return 1
        """
    )

    function = next(c for c in chunks if c.symbol_name == "lookup")
    assert function.start_line == 3
    assert function.code.startswith("# Cache the lookup.\n@functools.cache\ndef lookup():")
    assert_span_invariants(content, chunks)


def test_module_statements_between_definitions_keep_exact_spans():
    content, chunks = chunks_for(
        """
        import os

        def first():
            return 1

        if os.environ.get("X"):
            FLAG = True

        def second():
            return 2
        """
    )

    module_chunks = [c for c in chunks if c.symbol_name == MODULE_SYMBOL]
    assert [(c.start_line, c.end_line) for c in module_chunks] == [(1, 1), (6, 7)]
    assert "if os.environ.get" in module_chunks[1].code
    assert_span_invariants(content, chunks)


def test_large_class_splits_into_header_statements_and_methods_with_own_spans():
    methods = "\n".join(
        f"    def method_{index}(self):\n        return {index}  # {'x' * 40}\n"
        for index in range(6)
    )
    source = 'class Big(Base):\n    """Docstring."""\n    limit = 3\n\n' + methods
    content, chunks = chunks_for(source, max_chars=300)

    header = chunks[0]
    assert (header.symbol_name, header.symbol_type) == ("Big", "class")
    assert header.start_line == 1
    assert "class Big(Base):" in header.code and "limit = 3" in header.code

    method = next(c for c in chunks if c.symbol_name == "Big.method_3")
    assert method.symbol_type == "method"
    assert method.code.lstrip().startswith("def method_3(self):")
    # The citation span is the method's own lines, never the class start.
    assert method.start_line > header.end_line
    assert method.end_line - method.start_line == 1
    assert method.context == "class Big(Base):"
    assert_span_invariants(content, chunks)


def test_oversized_function_splits_at_parameters_and_body_statements():
    params = ",\n".join(f"    param_{index}: int = {index}" for index in range(12))
    body = "\n".join(f"    value_{index} = param_{index} * 2" for index in range(12))
    source = f"def wide(\n{params},\n) -> int:\n{body}\n    return value_0\n"
    content, chunks = chunks_for(source, max_chars=160)

    parts = [c for c in chunks if c.symbol_name == "wide"]
    assert len(parts) > 2
    assert [c.part for c in parts] == list(range(1, len(parts) + 1))
    assert all(c.part_count == len(parts) for c in parts)
    assert parts[0].code.startswith("def wide(")
    assert parts[-1].code.rstrip().endswith("return value_0")
    assert all(c.context == "def wide(" for c in parts[1:])
    # A parameter and its default are never separated.
    for chunk in parts:
        for line in chunk.code.splitlines():
            if "param_" in line and ":" in line:
                assert "=" in line
    assert_span_invariants(content, chunks)


def test_oversized_nested_function_splits_into_its_statements():
    inner = "\n".join(f"        step_{index} = {index}  # {'y' * 30}" for index in range(10))
    source = f"def outer():\n    async def app(request):\n{inner}\n        return step_0\n    return app\n"
    content, chunks = chunks_for(source, max_chars=200)

    parts = [c for c in chunks if c.symbol_name == "outer"]
    assert len(parts) >= 3
    assert all(len(c.code) <= 200 for c in parts)
    assert_span_invariants(content, chunks)


def test_single_statement_larger_than_budget_is_kept_whole():
    source = 'def f():\n    return "' + "z" * 500 + '"\n'
    content, chunks = chunks_for(source, max_chars=100)

    assert [c.code.strip() for c in chunks][0] == "def f():"
    statement = chunks[1]
    assert (statement.part, statement.part_count) == (2, 2)
    assert statement.code == content.splitlines()[1]
    assert len(statement.code) > 100
    assert_span_invariants(content, chunks)


def test_type_alias_is_a_module_statement():
    content, chunks = chunks_for(
        """
        type RouteMap = dict[str, str]

        def build_route_map():
            return {}
        """
    )

    assert chunks[0].symbol_name == MODULE_SYMBOL
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 1)
    assert_span_invariants(content, chunks)


def test_embedding_text_carries_location_context_and_code():
    methods = "\n".join(f"    def m{index}(self):\n        return {index}\n" for index in range(8))
    _, chunks = chunks_for("class Router:\n" + methods, max_chars=60)

    method = next(c for c in chunks if c.symbol_name == "Router.m2")
    text = method.embedding_text()
    assert text.splitlines()[0] == "fastapi/test.py :: Router.m2"
    assert "class Router:" in text
    assert text.endswith(method.code)


def test_unparseable_file_yields_no_chunks():
    assert chunk_file("bad.py", "fastapi/bad.py", "def broken(:\n") == []
