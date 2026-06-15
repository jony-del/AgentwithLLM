from agent_core.session import _MAX_READ_FILE_STATE, SessionContext


def test_record_read_orders_newest_last():
    session = SessionContext()
    session.record_read("a", "A1")
    session.record_read("b", "B1")
    session.record_read("c", "C1")
    assert list(session.read_file_state) == ["a", "b", "c"]


def test_record_read_re_read_moves_to_end_and_updates_content():
    session = SessionContext()
    session.record_read("a", "A1")
    session.record_read("b", "B1")
    session.record_read("a", "A2")  # re-read a → newest
    assert list(session.read_file_state) == ["b", "a"]
    assert session.read_file_state["a"] == "A2"


def test_record_read_caps_size_evicting_oldest():
    session = SessionContext()
    for i in range(_MAX_READ_FILE_STATE + 5):
        session.record_read(f"f{i}", f"c{i}")
    assert len(session.read_file_state) == _MAX_READ_FILE_STATE
    keys = list(session.read_file_state)
    # The first 5 inserted should have been evicted; newest survives.
    assert keys[0] == "f5"
    assert keys[-1] == f"f{_MAX_READ_FILE_STATE + 4}"
