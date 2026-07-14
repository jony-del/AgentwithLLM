from agent_core.permissions import PermissionMode
from agent_core.terminal.permission_picker import PermissionPicker


def test_permission_picker_contains_all_modes_and_exact_labels() -> None:
    rows = PermissionPicker(PermissionMode.DEFAULT).rows()
    assert [row.mode.value for row in rows] == [
        "default",
        "acceptedits",
        "plan",
        "auto",
        "dontask",
        "bypass",
    ]
    labels = {row.mode.value: row.label for row in rows}
    assert labels["default"] == "manual mode on"
    assert labels["acceptedits"] == "accept edits on"
    assert labels["plan"] == "plan mode on"
    assert labels["auto"] == "auto mode on"


def test_permission_picker_navigation_wraps() -> None:
    picker = PermissionPicker(PermissionMode.DEFAULT)
    picker.up()
    assert picker.selection() is PermissionMode.BYPASS
    picker.down()
    assert picker.selection() is PermissionMode.DEFAULT
