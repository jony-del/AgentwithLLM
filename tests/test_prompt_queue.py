from agent_core.terminal.prompt_queue import PromptQueue, QueuePriority


def test_queue_priority_fifo_and_separate_message_identity() -> None:
    queue = PromptQueue()
    later = queue.enqueue("later", priority="later")
    next_one = queue.enqueue("next one")
    queue.enqueue("now", priority="now")
    next_two = queue.enqueue("next two")

    first = queue.pop_between_turn()
    assert [item.content for item in first] == ["now"]
    second = queue.pop_between_turn()
    assert [item.content for item in second] == ["next one", "next two"]
    messages = [item.to_message(delivery="between_turn") for item in second]
    assert [message.uuid for message in messages] == [next_one.uuid, next_two.uuid]
    assert all(message.metadata["queued_command"] for message in messages)
    assert queue.pop_between_turn() == [later]


def test_slash_command_is_a_fifo_barrier_and_never_drains_midturn() -> None:
    queue = PromptQueue()
    queue.enqueue("steer now")
    queue.enqueue("/status")
    queue.enqueue("after slash")

    assert [message.content for message in queue.drain_midturn()] == ["steer now"]
    assert [item.content for item in queue.pop_between_turn()] == ["/status"]
    assert [item.content for item in queue.pop_between_turn()] == ["after slash"]


def test_recall_removes_all_editable_items_and_joins_with_newlines() -> None:
    queue = PromptQueue()
    queue.enqueue("first")
    queue.enqueue("fixed", editable=False)
    queue.enqueue("second", priority=QueuePriority.LATER)

    assert queue.recall_editable() == "first\nsecond"
    assert [item.content for item in queue.snapshot()] == ["fixed"]
    assert queue.recall_editable() == ""
