from adapters.langgraph.runtime.observability import (
    decision_summary,
    response_summary,
    safety_summary,
    tool_input_summary,
    tool_output_summary,
)


def test_event_capture_application_summary_preserves_each_item_and_outcome() -> None:
    state = {
        "tool_context": {
            "proposed_action": {
                "tool_name": "event_capture",
                "arguments": {
                    "items": [
                        {"name": "chicken", "quantity": 200, "unit": "g"},
                        {"name": "rice", "quantity": 150, "unit": "g"},
                    ]
                },
            },
            "last_tool_result": {
                "tool_name": "event_capture",
                "status": "success",
                "events_emitted": [{"event_id": "event-1"}],
                "data": {
                    "items": [
                        {"name": "chicken", "quantity": 200, "unit": "g"},
                        {"name": "rice", "quantity": 150, "unit": "g"},
                    ]
                },
            },
        }
    }

    tool_input, input_attributes = tool_input_summary(state)
    tool_output, output_attributes = tool_output_summary(state)

    assert tool_input == "chicken 200 g; rice 150 g"
    assert tool_output == "Meal captured: chicken 200 g; rice 150 g"
    assert input_attributes["victus.tool.item_count"] == 2
    assert output_attributes["victus.tool.events_emitted"] == 1


def test_application_summaries_expose_decision_safety_and_final_response() -> None:
    safety_output, safety_attributes = safety_summary(
        {
            "safety": {
                "status": "blocked",
                "decision": "route_to_safety_triage",
                "severity": "high",
                "categories": ["self_harm"],
            }
        }
    )
    decision_output, decision_attributes = decision_summary(
        {"tool_context": {"proposed_action": {"tool_name": "evidence_retrieval"}}}
    )
    response_output, response_attributes = response_summary(
        {"response": {"mode": "final", "user_message": "Meal logged."}}
    )

    assert safety_output == "Safety blocked: self_harm"
    assert safety_attributes["victus.safety.severity"] == "high"
    assert decision_output == "Selected tool: evidence_retrieval"
    assert decision_attributes["victus.tool.name"] == "evidence_retrieval"
    assert response_output == "Meal logged."
    assert response_attributes["victus.response.mode"] == "final"
