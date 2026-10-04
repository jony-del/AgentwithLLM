import pytest

from agent_core.merge import MergeConflict, three_way_merge


def test_merge_disjoint_python_functions_preserves_crlf_and_comments():
    base = "# keep\r\ndef a():\r\n    return 1\r\n\r\ndef b():\r\n    return 2\r\n"
    current = base.replace("return 1", "return 10")
    incoming = base.replace("return 2", "return 20")
    assert three_way_merge(base, current, incoming, python=True) == base.replace("return 1", "return 10").replace("return 2", "return 20")


def test_same_function_even_disjoint_lines_is_an_ast_conflict():
    base = "def f():\n    x=1\n    y=2\n    return x+y\n"
    with pytest.raises(MergeConflict, match="symbol"):
        three_way_merge(base, base.replace("x=1", "x=10"), base.replace("y=2", "y=20"), python=True)


def test_disjoint_class_methods_merge_but_interface_conflicts():
    base = "class C:\n    def a(self):\n        return 1\n\n    def b(self):\n        return 2\n"
    assert "return 20" in three_way_merge(base, base.replace("return 1", "return 10"), base.replace("return 2", "return 20"), python=True)
    with pytest.raises(MergeConflict):
        three_way_merge(base, base.replace("class C:", "class C(object):"), base.replace("return 2", "return 20"), python=True)


@pytest.mark.parametrize("current,incoming", [("x\nb\n", "y\nb\n"), ("a\nx\nb\n", "a\ny\nb\n")])
def test_text_overlap_and_same_insertion_reject(current, incoming):
    with pytest.raises(MergeConflict):
        three_way_merge("a\nb\n", current, incoming)


def test_three_way_rejects_malformed_python():
    with pytest.raises(MergeConflict):
        three_way_merge("def a():\n    return 1\ndef b():\n    return 2\n", "def a(\n", "def b(\n", python=True)


def test_python_bom_survives_ast_guard_and_merge():
    base = "\ufeffdef a():\n    return 1\n\ndef b():\n    return 2\n"
    result = three_way_merge(base, base.replace("return 1", "return 10"), base.replace("return 2", "return 20"), python=True)
    assert result == base.replace("return 1", "return 10").replace("return 2", "return 20")
