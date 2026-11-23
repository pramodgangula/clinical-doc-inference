"""LangGraph wiring for the per-field workflow.

The nodes and routing live in workflow.py; this module only declares the graph.
"""

from __future__ import annotations

from .workflow import FieldState, FieldWorkflow


def build_field_graph(wf: FieldWorkflow):
    from langgraph.graph import END, START, StateGraph  # optional dependency

    graph = StateGraph(FieldState)
    graph.add_node("retrieve", wf.retrieve)
    graph.add_node("generate", wf.generate)
    graph.add_node("parse", wf.parse)
    graph.add_node("validate", wf.validate)
    graph.add_node("give_up", wf.give_up)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "parse")
    graph.add_conditional_edges(
        "parse",
        wf.route_after_parse,
        {"retry": "generate", "validate": "validate", "give_up": "give_up"},
    )
    graph.add_edge("validate", END)
    graph.add_edge("give_up", END)
    return graph.compile()


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True
