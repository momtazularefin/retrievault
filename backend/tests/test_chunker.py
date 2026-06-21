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
