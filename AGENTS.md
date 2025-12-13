# TianGong AI Workspace — Agent Guide

## Overview
- Unified developer workspace for coordinating Codex, Gemini, Claude Code, and both document-centric & autonomous LangGraph / DeepAgents workflows.
- Python 3.12+ project managed完全 by `uv`; avoid `pip`, `poetry`, `conda`.
- Primary entry point: `uv run tiangong-workspace`, featuring LangChain/LangGraph document agents, LangGraph planning agents, and Tavily MCP research.

## Repository Layout
- `src/tiangong_ai_workspace/cli.py`: Typer CLI with `docs`, `agents`, `research`, `knowledge`, `embeddings`, `citation-study`, `mineru-with-images`, and `mcp` subcommands plus structured JSON output support.
- `src/tiangong_ai_workspace/agents/`:
  - `workflows.py`: LangChain/LangGraph document workflows (reports, plans, patent, proposals).
  - `deep_agent.py`: Workspace autonomous agent supporting both native LangGraph loops and the `deepagents` runtime.
  - `tools.py`: LangChain Tool wrappers for shell/Python execution, Tavily search, Neo4j CRUD, and document generation (with typed Pydantic schemas).
- `src/tiangong_ai_workspace/tooling/`: Utilities shared by agents.
  - `responses.py`: `WorkspaceResponse` envelope for deterministic outputs.
  - `registry.py`: Tool metadata registry surfaced via `tiangong-workspace tools --catalog`.
  - `config.py`: Loads CLI/tool registry configuration from `pyproject.toml`.
  - `tool_schemas.py`: Pydantic schemas exported to LangChain tools and registry metadata.
  - `llm.py`: Provider-agnostic model router (OpenAI provider registered by default).
  - `embeddings.py`: OpenAI-compatible embedding client surfaced via CLI/registry.
  - `tavily.py`: Tavily MCP client with retry + structured payloads.
  - `dify.py`: Direct HTTP client for the Dify knowledge base (no MCP required).
  - `openalex.py`: OpenAlex client plus heuristic citation potential classifier.
  - `mineru.py`: HTTP client for the Mineru PDF image extraction API.
  - `neo4j.py`: Neo4j driver wrapper used by CRUD tools and registry metadata.
  - `executors.py`: Shell/Python execution helpers with timeouts, allow-lists, and structured telemetry for agent consumption.
- `src/tiangong_ai_workspace/templates/`: Markdown scaffolds referenced by workflows.
- `.sercrets/secrets.toml`: Local-only secrets (copy from `.sercrets/secrets.example.toml`).

## Workspace Configuration
- The `[tool.tiangong.workspace]` section inside `pyproject.toml` now controls detected CLI tools (`cli_tools`) and registry entries (`tool_registry`).
- Updating those tables automatically refreshes `tiangong-workspace tools`/`tools --catalog` without editing Python sources.
- Registry metadata is enriched with JSON schemas from `tooling.tool_schemas`, enabling downstream agents to understand tool inputs/outputs.

## Tooling Workflow
Run everything through `uv`:

```bash
uv sync
uv run tiangong-workspace --help
```

After **every** code change run, in order:

```bash
uv run black .
uv run ruff check
uv run pytest
```

All three must pass before sharing updates.

## CLI Quick Reference
- `uv run tiangong-workspace info` — workspace summary.
- `uv run tiangong-workspace check` — validate Python/uv/Node + registered CLIs.
- `uv run tiangong-workspace tools --catalog` — list internal agent workflows from the registry.
- `uv run tiangong-workspace docs list` — supported document workflows.
- `uv run tiangong-workspace docs run <workflow> --topic ...` — generate drafts (supports `--json`, `--skip-research`, `--purpose`, `--ai-review`, etc.).
- `uv run tiangong-workspace agents list` — view autonomous agents + runtime executors available to agents.
- `uv run tiangong-workspace agents run "<task>" [--no-shell/--no-python/--no-tavily/--no-dify/--no-document --engine langgraph|deepagents]` — run the workspace DeepAgent with the preferred backend.
- `uv run tiangong-workspace research "<query>"` — invoke Tavily MCP search (also supports `--json`).
- `uv run tiangong-workspace knowledge retrieve "<query>"` — call the Dify knowledge base API without MCP；可用 `--search-method`、`--reranking/--no-reranking`、`--reranking-provider/--reranking-model`、`--score-threshold`、`--semantic-weight` 与 `--metadata` 快速配置 Dify `retrieval_model` 与元数据过滤。
- `uv run tiangong-workspace embeddings generate "<text>"` — 调用 OpenAI 兼容 embedding 服务，支持批量文本、`--model/--json`。
- `uv run tiangong-workspace citation-study "large language model" --since-year 2020 --limit 30` — 调用 OpenAlex 获取近年论文并按高/中/低引用潜力分类。
- `uv run tiangong-workspace mineru-with-images ./file.pdf --prompt "解析图表"` — 调用 Mineru PDF 图片解析 API，支持 MinIO 落盘与模型透传。
- `uv run tiangong-workspace mcp services|tools|invoke` — inspect and call configured MCP services.

Use `--json` for machine-readable responses suitable for chaining agents.

## Secrets
- Populate `.sercrets/secrets.toml` using the example file.
- Required: `openai.api_key`. Optional: `model`, `chat_model`, `deep_research_model`.
- Tavily section needs `service_name`, `url`, and `api_key` (`Authorization: Bearer` header).
- Neo4j section (optional) defines `uri`, `username`, `password`, and `database`; when absent the Neo4j LangChain tool is automatically disabled.
- `dify_knowledge_base` defines `api_base_url`, `api_key`, and `dataset_id`; this powers the `knowledge retrieve` CLI and the LangChain Dify tool (no MCP block required).
- `openai_compatitble_embedding` defines `url`, optional `api_key`, and `model` for the embedding CLI/registry tool；`api_key` 可留空以兼容无鉴权服务。
- `mineru` defines `api_url` and `token` for the Mineru PDF 图片解析服务；CLI 支持 `--url`/`--token` 覆盖默认值。
- Secrets stay local; never commit `.sercrets/`.

## Maintenance Rules
- Modify program code → update both `AGENTS.md` and `README.md`.
- Respect dependency declarations in `pyproject.toml`; use `uv add/remove`.
- Prefer ASCII in source files unless the file already uses other encodings.
- Structured outputs (`WorkspaceResponse`) keep agent integrations predictable—adhere to them when adding new commands.

## Helpful Notes
- To stub LLM calls in tests, inject a custom `Runnable` when calling `run_document_workflow`.
- Tavily wrapper retries transient failures; propagate explicit `TavilySearchError` for agents to handle.
- Register new workflows via `tooling.registry.register_tool` for discoverability.
- Shell/Python executors enforce configurable timeouts and command allow-lists—reuse them instead of invoking `subprocess` or `exec` directly.
- LangChain tools should depend on the schemas in `tooling.tool_schemas` so registry metadata stays consistent.
- Neo4j automation lives in `tooling.neo4j`; reuse `Neo4jClient` + `Neo4jCommand*` schemas to expose graph operations or add migrations/tests.
- Dify knowledge base access lives in `tooling.dify` and `agents/tools.create_dify_knowledge_tool`; reuse them to expose retrieval without MCP transport，必要时直接使用 `RetrievalModelConfig`、`MetadataFilterGroup` 等帮助类来构建与 API 规范一致的检索请求。
- Choose the DeepAgents backend via `--engine deepagents` when you need its filesystem/todo middleware; ensure the supplied LLM implements `BaseChatModel`.
- Keep logs redaction-aware if adding persistence; avoid leaking API keys.
- Workspace agent factory accepts `model`, `include_*` flags, and additional tools/subagents. Reuse `tooling.executors` or extend `agents/tools.py` when exposing new capabilities to autonomous agents.
