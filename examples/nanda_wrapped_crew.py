#!/usr/bin/env python3
# Minimal Nanda wrapper around your HW1 CrewAI pipeline.
import os, sys
from contextlib import closing
from typing import Type
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM

# Optional web search tool, with Serper -> DDGS fallback like HW1
web_tool = None
try:
    from crewai_tools import SerperDevTool  # type: ignore
    if os.getenv("SERPER_API_KEY"):
        web_tool = SerperDevTool()
except Exception:
    web_tool = None

if web_tool is None:
    try:
        from ddgs import DDGS
        from crewai.tools import BaseTool
        class WebSearchInput(BaseModel):
            query: str
            max_results: int = 5
        class WebSearchTool(BaseTool):
            name: str = "Web Search"
            description: str = "Search the web and return up to N results with title and url."
            args_schema: Type[BaseModel] = WebSearchInput
            def _run(self, query: str, max_results: int = 5) -> str:
                rows = []
                with closing(DDGS()) as ddgs:
                    for i, r in enumerate(ddgs.text(query, max_results=max_results), start=1):
                        title = r.get("title", "No title")
                        href = r.get("href", "")
                        rows.append(f"{i}. {title} - {href}")
                return "\n".join(rows)
        web_tool = WebSearchTool()
    except Exception:
        web_tool = None

# File writer fallback
file_writer_tool = None
try:
    from crewai_tools import FileWriterTool  # type: ignore
    file_writer_tool = FileWriterTool()
except Exception:
    from crewai.tools import BaseTool
    class FileWriterInput(BaseModel):
        path: str
        content: str
    class FileWriterToolFallback(BaseTool):
        name: str = "File Writer"
        description: str = "Write given text content to a file path."
        args_schema: Type[BaseModel] = FileWriterInput
        def _run(self, path: str, content: str) -> str:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path}"
    file_writer_tool = FileWriterToolFallback()

PERSONA = (
    "You are Tzuri's digital twin lite - MIT Sloan MBA Class of 2026, former Israeli Navy SEAL team leader, "
    "Tesla Service Operations intern, and RA to Prof. Andrew McAfee. "
    "Strengths: structured research, operations thinking, data informed decisions, crisp writing, networking follow ups. "
    "Communication: clear, direct, friendly, no em dash - use hyphen."
)

def llm():
    model_name = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
    return LLM(model=model_name, temperature=0.3, max_tokens=800)


def create_research_agent():
    tools = [t for t in [web_tool] if t]
    return Agent(
        role="Digital Twin Lite - Research Assistant",
        goal="Run focused research sprints and outline WRDS or FactSet pulls.",
        backstory=PERSONA,
        verbose=True, allow_delegation=False, tools=tools, llm=llm()
    )

def create_writer_agent():
    return Agent(
        role="Digital Twin Lite - Writer",
        goal="Turn notes into concise Markdown and save to file. Emit runnable pandas or matplotlib when asked.",
        backstory=PERSONA,
        verbose=True, allow_delegation=False, tools=[file_writer_tool], llm=llm()
    )

def create_editor_agent():
    tools = [t for t in [file_writer_tool, web_tool] if t]
    return Agent(
        role="Digital Twin Lite - Editor and Fact Checker",
        goal="Improve clarity and persona fit. Replace em dash with hyphen. Save final file.",
        backstory=PERSONA,
        verbose=True, allow_delegation=False, tools=tools, llm=llm()
    )

def create_research_task(agent, topic: str):
    return Task(
        description=f"Research the topic: {topic}\nSummarize concepts, trends, benefits, challenges, examples.",
        expected_output="Structured research summary with examples and sources list.",
        agent=agent
    )

def create_writing_task(agent, topic: str):
    return Task(
        description=f"Write an 800-1000 word article about: {topic}\nSave to ai_studio_article.md",
        expected_output="Article saved as ai_studio_article.md",
        agent=agent
    )

def create_editing_task(agent):
    return Task(
        description=("Improve the article, ensure hyphen instead of em dash, add Edits applied section, "
                     "and save as ai_studio_article_reviewed.md."),
        expected_output="Improved Markdown saved as ai_studio_article_reviewed.md.",
        agent=agent
    )

def run_crew_for_topic(topic: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set."
    researcher = create_research_agent()
    writer = create_writer_agent()
    editor = create_editor_agent()
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[create_research_task(researcher, topic),
               create_writing_task(writer, topic),
               create_editing_task(editor)],
        process=Process.sequential, verbose=True
    )
    try:
        result = crew.kickoff()
        final_path = "ai_studio_article_reviewed.md"
        if os.path.exists(final_path):
            with open(final_path, "r", encoding="utf-8") as f:
                return f.read()
        return str(result)
    except Exception as e:
        return f"Error running Crew pipeline: {e}"

from nanda_adapter import NANDA
def create_improvement_from_crew():
    def improve(message_text: str) -> str:
        topic = (message_text or "").strip() or "Artificial Intelligence in Education"
        out = run_crew_for_topic(topic)
        # Force a clean string response for the adapter
        if out is None:
            return "No output produced."
        if isinstance(out, bytes):
            try:
                return out.decode("utf-8", errors="replace")
            except Exception:
                return str(out)
        return str(out)
    return improve

def main():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    domain = os.getenv("DOMAIN_NAME")
    if not anthropic_key or not domain:
        print("Missing env vars. Set ANTHROPIC_API_KEY and DOMAIN_NAME.")
        sys.exit(1)
    nanda = NANDA(create_improvement_from_crew())
    nanda.start_server_api(anthropic_key, domain)

if __name__ == "__main__":
    main()