"""
Workspace-specialised autonomous agent built with LangGraph.

The agent coordinates shell execution, Python scripting, Tavily research, and
document drafting tools. It follows a lightweight ReAct-style loop that plans an
action, executes the selected tool, and reasons over observations until it
produces a final answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TypedDict

from deepagents import create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph

from ..tooling.dify import DifyKnowledgeBaseError
from ..tooling.llm import ModelRouter
from ..tooling.neo4j import Neo4jToolError
from ..tooling.tavily import TavilySearchClient, TavilySearchError
from .tools import (
    create_crossref_tool,
    create_dify_knowledge_tool,
    create_document_tool,
    create_neo4j_tool,
    create_openalex_tools,
    create_python_tool,
    create_shell_tool,
    create_tavily_tool,
)

__all__ = ["build_workspace_deep_agent", "WorkspaceAgentConfig"]


class WorkspaceAgentState(TypedDict, total=False):
    """State managed by the LangGraph-powered workspace agent."""

    messages: list[BaseMessage]
    iterations: int
    action: str
    action_input: Any
    thought: str
    last_observation: str
    final_response: str


@dataclass(slots=True)
class WorkspaceAgentConfig:
    """Configuration options for the workspace agent."""

    max_iterations: int = 8
    system_prompt: str | None = None


_TOOL_SENTINEL = "Available tools: shell, python, tavily, crossref, openalex_work, openalex_cited_by, document, neo4j, knowledge."

DEFAULT_SYSTEM_PROMPT = f"""You are the TianGong Workspace orchestrator.
- Plan multi-step solutions and choose the best tool for each step.
- {_TOOL_SENTINEL} Use shell/python for code or CLI tasks, tavily for web research, document for structured drafts.
- Think step-by-step. When you decide to finish, return a concise, helpful summary of the work performed.
- Always respond using JSON with keys: thought, action, input, final_response (only set when action is \"finish\")."""

TOOL_FALLBACK_MESSAGE = "An internal error occurred while running tool '{action}'. Include any partial results and continue."
SUPPORTED_ENGINES = {"langgraph", "deepagents"}


def build_workspace_deep_agent(
    *,
    model: Any | None = None,
    llm: Runnable | None = None,
    include_shell: bool = True,
    include_python: bool = True,
    include_tavily: bool = True,
    include_dify_knowledge: bool = True,
    include_document_agent: bool = True,
    include_neo4j: bool = True,
    include_crossref: bool = True,
    include_openalex: bool = True,
    system_prompt: str | None = None,
    max_iterations: int = 8,
    engine: str = "langgraph",
) -> Any:
    """
    Construct and compile the workspace autonomous agent.

    Parameters
    ----------
    model:
        Optional language model (LangChain-compatible) or provider identifier. When
        omitted and `llm` is not provided, the default OpenAI configuration is used.
    llm:
        Optional runnable to override the planning model (primarily for testing).
    include_*:
        Flags to toggle the availability of individual tools (shell, python, tavily, crossref, openalex, knowledge, document, neo4j).
    system_prompt:
        Custom system instructions appended to the default planner guidance.
    max_iterations:
        Safety limit on the number of planning cycles before the agent returns.
    """

    planner_llm = _resolve_planner_llm(llm=llm, model=model)
    config = WorkspaceAgentConfig(max_iterations=max_iterations, system_prompt=system_prompt)

    tools = _initialise_tools(
        include_shell=include_shell,
        include_python=include_python,
        include_tavily=include_tavily,
        include_document_agent=include_document_agent,
        include_dify_knowledge=include_dify_knowledge,
        include_neo4j=include_neo4j,
        include_crossref=include_crossref,
        include_openalex=include_openalex,
    )

    engine_choice = engine.lower().strip()
    if engine_choice not in SUPPORTED_ENGINES:
        available = ", ".join(sorted(SUPPORTED_ENGINES))
        raise ValueError(f"Unsupported agent engine '{engine}'. Available engines: {available}.")

    tool_list = _describe_tools(tools)

    if engine_choice == "deepagents":
        chat_model = _require_chat_model(planner_llm)
        return _build_deepagents_agent(chat_model, tools, config, tool_list)

    return _build_langgraph_agent(planner_llm, tools, config, tool_list)


def _initialise_tools(
    *,
    include_shell: bool,
    include_python: bool,
    include_tavily: bool,
    include_document_agent: bool,
    include_dify_knowledge: bool,
    include_neo4j: bool,
    include_crossref: bool,
    include_openalex: bool,
) -> Mapping[str, Any]:
    tool_mapping: MutableMapping[str, Any] = {}

    if include_shell:
        tool_mapping["shell"] = create_shell_tool()
    if include_python:
        tool_mapping["python"] = create_python_tool()
    if include_tavily:
        try:
            TavilySearchClient()  # validate configuration early
            tool_mapping["tavily"] = create_tavily_tool()
        except TavilySearchError:
            # Skip Tavily if secrets are missing; agent can still operate offline.
            pass
    if include_dify_knowledge:
        try:
            tool_mapping["knowledge"] = create_dify_knowledge_tool()
        except DifyKnowledgeBaseError:
            pass
    if include_document_agent:
        tool_mapping["document"] = create_document_tool()
    if include_neo4j:
        try:
            tool_mapping["neo4j"] = create_neo4j_tool()
        except Neo4jToolError:
            pass
    if include_crossref:
        tool_mapping["crossref"] = create_crossref_tool()
    if include_openalex:
        work_tool, cited_by_tool = create_openalex_tools()
        tool_mapping["openalex_work"] = work_tool
        tool_mapping["openalex_cited_by"] = cited_by_tool

    return tool_mapping


def _resolve_planner_llm(*, llm: Runnable | None, model: Any | None) -> Runnable:
    if llm is not None:
        return llm

    if hasattr(model, "invoke"):
        return model  # Already a LangChain runnable

    factory = ModelRouter()
    if isinstance(model, str):
        return factory.create_chat_model(model_override=model, temperature=0.3)

    return factory.create_chat_model(purpose="deep_research", temperature=0.3)


def _build_langgraph_agent(
    planner_llm: Runnable,
    tools: Mapping[str, Any],
    config: WorkspaceAgentConfig,
    tool_list: str,
) -> Any:
    planner = _build_planner_chain(planner_llm, tools, config, tool_list)

    graph = StateGraph(WorkspaceAgentState)
    graph.add_node("plan", _make_plan_node(planner, config, tools))
    graph.add_node("act", _make_action_node(tools))
    graph.set_entry_point("plan")

    graph.add_conditional_edges("plan", _make_plan_router(tools, config))
    graph.add_edge("act", "plan")

    return graph.compile()


def _build_deepagents_agent(
    planner_llm: BaseChatModel,
    tools: Mapping[str, Any],
    config: WorkspaceAgentConfig,
    tool_list: str,
) -> Any:
    system_prompt = _compose_system_prompt(tool_list, config.system_prompt)
    return create_deep_agent(
        model=planner_llm,
        tools=list(tools.values()),
        system_prompt=system_prompt,
    )


def _describe_tools(tools: Mapping[str, Any]) -> str:
    if not tools:
        return "- finish: provide the final answer (no tools available)."
    lines = []
    for name, tool in tools.items():
        description = getattr(tool, "description", "") or "No description."
        lines.append(f"- {name}: {description}")
    return "\n".join(lines)


def _require_chat_model(model: Runnable) -> BaseChatModel:
    if isinstance(model, BaseChatModel):
        return model
    raise ValueError("The deepagents engine requires a LangChain BaseChatModel-compatible llm.")


def _build_planner_chain(
    planner_llm: Runnable,
    tools: Mapping[str, Any],
    config: WorkspaceAgentConfig,
    tool_list: str,
) -> Runnable:

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _compose_system_prompt(tool_list, config.system_prompt)),
            (
                "human",
                (
                    "Conversation so far:\n{history}\n\n"
                    "When deciding on the next action, respond with strict JSON:\n"
                    "```json\n"
                    "{{\n"
                    '  "thought": "<reasoning>",\n'
                    '  "action": "<tool or finish>",\n'
                    '  "input": <arguments>,\n'
                    '  "final_response": "<only when action is finish>"\n'
                    "}}\n"
                    "```\n"
                    "If you choose a tool, ensure `input` matches its expected parameters."
                ),
            ),
        ]
    )

    return prompt | planner_llm | StrOutputParser()


def _compose_system_prompt(tool_list: str, custom_prompt: str | None) -> str:
    base = DEFAULT_SYSTEM_PROMPT.replace(_TOOL_SENTINEL, f"Available tools:\n{tool_list}")
    if custom_prompt:
        return f"{custom_prompt.strip()}\n\n{base}"
    return base


def _make_plan_node(
    planner: Runnable,
    config: WorkspaceAgentConfig,
    tools: Mapping[str, Any],
) -> Callable[[WorkspaceAgentState], WorkspaceAgentState]:
    def plan_node(state: WorkspaceAgentState) -> WorkspaceAgentState:
        messages = list(state.get("messages", []))
        history_text = _render_history(messages)

        response_text = planner.invoke({"history": history_text})
        plan = _parse_plan(response_text)

        iterations = state.get("iterations", 0) + 1
        action = plan.get("action", "finish")
        action_input = plan.get("input")
        thought = plan.get("thought") or response_text

        if action not in {*tools.keys(), "finish"}:
            action = "finish"
            plan["final_response"] = plan.get("final_response") or f"Unsupported action '{plan.get('action')}'. Provide the best possible summary instead."

        messages.append(AIMessage(content=f"Thought: {thought}\nAction: {action}"))

        new_state: WorkspaceAgentState = {
            **state,
            "messages": messages,
            "iterations": iterations,
            "action": action,
            "action_input": action_input,
            "thought": thought,
        }

        if action == "finish" or iterations >= config.max_iterations:
            final_text = plan.get("final_response")
            if not final_text:
                final_text = plan.get("message") or plan.get("input") or thought
            new_state["final_response"] = str(final_text)

        return new_state

    return plan_node


def _make_action_node(tools: Mapping[str, Any]) -> Callable[[WorkspaceAgentState], WorkspaceAgentState]:
    def act_node(state: WorkspaceAgentState) -> WorkspaceAgentState:
        action = state.get("action")
        if not action:
            return state

        tool = tools.get(action)
        if tool is None:
            return {
                **state,
                "final_response": f"No tool named '{action}' is available. Summarise progress and stop.",
                "action": "finish",
            }

        action_input = state.get("action_input")
        invocation = _normalise_tool_input(action_input)

        try:
            result = tool.invoke(invocation)
        except Exception as exc:  # pragma: no cover - defensive fallback
            observation = TOOL_FALLBACK_MESSAGE.format(action=action) + f" Error: {exc}"
        else:
            observation = _render_observation(result)

        messages = list(state.get("messages", []))
        messages.append(ToolMessage(content=observation, name=action))

        return {
            **state,
            "messages": messages,
            "action": None,
            "action_input": None,
            "last_observation": observation,
        }

    return act_node


def _make_plan_router(
    tools: Mapping[str, Any],
    config: WorkspaceAgentConfig,
) -> Callable[[WorkspaceAgentState], str]:
    def router(state: WorkspaceAgentState) -> str:
        action = state.get("action")
        iterations = state.get("iterations", 0)

        if action == "finish" or iterations >= config.max_iterations:
            return END
        if action not in tools:
            return END
        return "act"

    return router


def _normalise_tool_input(action_input: Any) -> Any:
    if action_input is None:
        return {}
    if isinstance(action_input, str):
        return action_input
    if isinstance(action_input, Mapping):
        return dict(action_input)
    return action_input


def _render_observation(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except TypeError:
        return str(result)


def _render_history(messages: Sequence[BaseMessage]) -> str:
    rendered: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = type(message).__name__.replace("Message", "").lower()
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        rendered.append(f"{index}. [{role}] {content}")
    return "\n".join(rendered) if rendered else "(no prior assistant actions)"


def _parse_plan(text: str) -> MutableMapping[str, Any]:
    candidate = text.strip()

    if "```" in candidate:
        segments = []
        for block in candidate.split("```"):
            block = block.strip()
            if not block:
                continue
            if block.lower().startswith("json"):
                block = block[4:].strip()
            segments.append(block)
        if segments:
            candidate = segments[0]

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, Mapping):
            return dict(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback: treat raw text as final response
    return {"action": "finish", "final_response": candidate}
