from retrievault.chunker import chunk_file


def test_chunk_file_extracts_functions_and_classes():
    content = """
import os

def my_func():
    return 1

class MyClass:
    def method(self):
        pass
"""
    chunks = chunk_file("test.py", "fastapi/test.py", content)
    
    # Expect 1 module preamble, 1 function, 1 class (since it's small)
    assert len(chunks) == 3
    
    preamble = chunks[0]
    assert preamble.symbol_name == "__module_preamble__"
    assert preamble.symbol_type == "module"
    assert "import os" in preamble.code
    
    func = chunks[1]
    assert func.symbol_name == "my_func"
    assert func.symbol_type == "function"
    assert func.start_line == 4
    
    cls = chunks[2]
    assert cls.symbol_name == "MyClass"
    assert cls.symbol_type == "class"
    assert cls.start_line == 7


def test_chunk_file_includes_type_aliases_in_module_preamble():
    content = """type RouteMap = dict[str, str]

def build_route_map():
    return {}
"""

    chunks = chunk_file("routes.py", "fastapi/routes.py", content)

    preamble = chunks[0]
    assert preamble.symbol_name == "__module_preamble__"
    assert preamble.symbol_type == "module"
    assert preamble.start_line == 1
    assert preamble.end_line == 1
    assert "type RouteMap = dict[str, str]" in preamble.code


def test_chunk_file_splits_large_class_methods_with_matching_header_span():
    filler = "\n".join(f"    attr_{index} = {index}" for index in range(101))
    content = f"""class BigClass:
{filler}
    def method(self):
        return self.attr_0
"""

    chunks = chunk_file("big.py", "fastapi/big.py", content)

    assert len(chunks) == 1
    method = chunks[0]
    assert method.symbol_name == "BigClass.method"
    assert method.symbol_type == "method"
    assert method.start_line == 1
    assert method.end_line == 104
    assert method.code.startswith("class BigClass:\n")
    assert "    def method(self):" in method.code
