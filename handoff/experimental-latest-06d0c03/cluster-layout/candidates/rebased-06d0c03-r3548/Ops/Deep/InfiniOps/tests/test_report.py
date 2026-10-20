from types import SimpleNamespace

from tests import report


def _item(module_name, params=None):
    return SimpleNamespace(
        module=SimpleNamespace(__name__=f"tests.{module_name}"),
        callspec=SimpleNamespace(params=params or {}),
        nodeid=f"tests/{module_name}.py::test_case",
        location=(f"tests/{module_name}.py", 0, "test_case"),
        originalname="test_case",
        name="test_case",
    )


def test_item_context_infers_native_operator_modules():
    context = report._item_context(_item("test_add", {"device": "cuda"}))

    assert context["operator"] == "add"
    assert context["aten_name"] == "add"
    assert context["torch_device"] == "cuda"


def test_item_context_does_not_classify_helper_modules_as_operators():
    context = report._item_context(_item("test_generate_wrappers"))

    assert context["operator"] is None
    assert context["aten_name"] is None
