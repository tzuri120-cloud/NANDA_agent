#!/usr/bin/env python3
"""
Nanda-wrapped version of your MIT AI Studio - Crew AI Example Agent.

Behavior:
- Each inbound message text is treated as the topic for your Crew pipeline.
- Runs Research -> Write -> Edit flow from your HW1 logic.
- Returns the final reviewed article text as the response.

Requires:
- ANTHROPIC_API_KEY in env for CrewAI LLM
- DOMAIN_NAME in env for the Nanda HTTPS client API
- fullchain.pem and privkey.pem present in the current folder
"""

import os
import sys
from typing import Optional, Type
from contextlib import closing

# --- Your HW1 dependencies ---
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel

# Optional web tool, same safe fallbacks you used
web_tool = None
try:
    from crewai_tools import SerperDevTool  # type: ignore
    if os.getenv("SERPER_API_KEY"):
        web_tool = SerperDevTool()
except Exception:
    web_tool = None

if web_tool is None:
    try:
        # pip install ddgs
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

# File writer tool fallback, like in HW1
file_writer_tool = None
try:
    from crewai_tools import FileWriterTool  # type: ignore
    file_writer_tool = FileWriterTool()
except Exception:
    from crewai.tools import BaseTool  # fallback

    class FileWriterInput(BaseModel):
        path: str
        content: str

    class FileWriterToolFallback(BaseTool):
        name: str = "File Writer"
        description: str = "Write given text content to a file path."
        args_schema: Type[BaseModel] = FileWriterInput

        def _run(self, path: str, content: str) -> str:
            import os as _os
            _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} chars to {path}"

    file_writer_tool = FileWriterToolFallback()

# Persona and agents - adapted from your HW1 file
PERSONA = (
    "You are Tzuri's digital twin lite - MIT Sloan MBA Class of 2026, former Israeli Navy SEAL team leader, "
    "Tesla Service Operations intern, and current Research Assistant to Prof. Andrew McAfee at MIT. "
    "Strengths: structured research, operations thinking, data informed decisions, crisp writing, networking follow ups. "
    "Communication: clear, direct, friendly, no em dash - use hyphen. "
    "What you handle on my behalf:\n"
    "- RA workflows - scope and run small research sprints, gather data from WRDS, FactSet, Compustat, CRSP, and public filings; draft reproducible steps; maintain data dictionaries and logs\n"
    "- Data pulls - produce sample Python or SQL queries for WRDS and FactSet, clean with pandas, save CSVs in organized folders\n"
    "- Visualization - generate ready to run matplotlib code to create charts and save PNGs\n"
    "- Rapid web research and credible source triage with citations\n"
    "- Summaries of cases, papers, and readings in tight bullets with action items\n"
    "- Draft concise emails to recruiters, professors, and partners in a casual professional tone\n"
    "- Create short memos, outlines, and Markdown docs, then save via the File Writer tool\n"
    "- Generate follow up checklists and next steps for projects, interviews, and internships\n"
    "Constraints: be brief, prefer bullets, include concrete dates and numbers when known, respect privacy."
)

def _llm():
    # Your HW1 default: pick up ANTHROPIC_API_KEY via LLM(model=...)
    model_name = os.getenv("LLM_MODEL", "ANTHROPIC_API_KEY")
    return LLM(model=model_name, temperature=0.3, max_tokens=800)

def create_research_agent():
    tools = [t for t in [web_tool] if t]
    return Agent(
        role="Digital Twin Lite - Research Assistant",
        goal=(
            "Run focused research sprints, gather credible facts and datasets, "
            "plan WRDS and FactSet pulls with sample queries, and outline pandas cleaning and matplotlib visualizations."
        ),
        backstory=PERSONA,
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=_llm(),
    )

def create_writer_agent():
    return Agent(
        role="Digital Twin Lite - Writer",
        goal=(
            "Turn notes and data into concise memos, emails, and Markdown articles and save them to files. "
            "When asked, emit runnable Python snippets for pandas and matplotlib."
        ),
        backstory=PERSONA,
        verbose=True,
        allow_delegation=False,
        tools=[file_writer_tool],
        llm=_llm(),
    )

def create_editor_agent():
    tools = [t for t in [file_writer_tool, web_tool] if t]
    return Agent(
        role="Digital Twin Lite - Editor and Fact Checker",
        goal=(
            "Review and refine drafts for clarity, structure, and persona fit. "
            "Fact check key claims when a web tool is available. "
            "Ensure no em dash is used - replace with hyphen. "
            "Save the improved version to a new file."
        ),
        backstory=PERSONA,
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=_llm(),
    )

def create_research_task(agent, topic: str):
    return Task(
        description=f"""Research the topic: {topic}

Your task is to:
1. Gather information about the key concepts and recent developments
2. Identify the main benefits and challenges
3. Find relevant examples or case studies
4. Summarize your findings in a structured format

Focus on credible sources and current information.""",
        expected_output="""A comprehensive research summary containing:
- Key concepts and definitions
- Recent developments and trends
- Benefits and challenges
- Relevant examples
- List of sources used""",
        agent=agent
    )

def create_writing_task(agent, topic: str):
    return Task(
        description=f"""Write a comprehensive article about: {topic}

Your task is to:
1. Write an engaging introduction to the topic
2. Explain key concepts and their importance
3. Discuss current trends and developments
4. Include practical examples or applications
5. Conclude with insights about the future
6. Use File Writer to save the article to 'ai_studio_article.md'

Make the article informative yet accessible, around 800 to 1000 words.""",
        expected_output="""A well written article saved as 'ai_studio_article.md' with:
- Engaging introduction
- Clear explanations of key concepts
- Current trends and developments
- Practical examples
- Future insights
- Proper Markdown formatting""",
        agent=agent
    )

def create_editing_task(agent):
    return Task(
        description="""
You are given the article just produced in the previous task context.
Improve clarity, structure, tone, and alignment with Tzuri's persona.
If a web tool is available, fact check 2 to 3 central claims and adjust wording if needed.
Replace any em dash with a hyphen.
Append a short "Edits applied" section with bullet points and, if fact checking was possible, a "Sources checked" list.
Use File Writer to save the final result as 'ai_studio_article_reviewed.md'.
If the full article text is not available in context, regenerate a clean article on the same topic and still save it.
""".strip(),
        expected_output="Improved Markdown saved as ai_studio_article_reviewed.md.",
        agent=agent,
    )

def run_crew_for_topic(topic: str) -> str:
    """Run your HW1 Crew pipeline for the given topic and return final text."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set. Export it before running."

    researcher = create_research_agent()
    writer = create_writer_agent()
    editor = create_editor_agent()

    research_task = create_research_task(researcher, topic)
    writing_task = create_writing_task(writer, topic)
    editing_task = create_editing_task(editor)

    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        # Prefer the reviewed file content if present
        final_path = "ai_studio_article_reviewed.md"
        if os.path.exists(final_path):
            with open(final_path, "r", encoding="utf-8") as f:
                return f.read()
        return str(result)
    except Exception as e:
        return f"Error running Crew pipeline: {e}"

# --- Nanda Adapter glue ---
from nanda_adapter import NANDA

def create_improvement_from_crew():
    """Map inbound message text to your Crew pipeline."""
    def improve(message_text: str) -> str:
        topic = message_text.strip() or "Artificial Intelligence in Education"
        return run_crew_for_topic(topic)
    return improve

def main():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    domain = os.getenv("DOMAIN_NAME")
    if not anthropic_key or not domain:
        print("Missing env vars. Set ANTHROPIC_API_KEY and DOMAIN_NAME.")
        sys.exit(1)

    nanda = NANDA(create_improvement_from_crew())
    # This launches:
    # - Agent bridge on port 6000
    # - HTTPS Flask API on port 6001 using ./fullchain.pem and ./privkey.pem
    nanda.start_server_api(anthropic_key, domain)

if __name__ == "__main__":
    main()
