"""MCP server exposing each agent as an independent tool.

Run it with:
    python -m mcp_server.server

You can then plug this server into Claude Desktop or any MCP-compatible client
by pointing the client's MCP config at this script. Each agent becomes a
callable tool with a typed JSON schema.

Tools:
    analyze_jd(jd_text)              -> JDAnalysis
    analyze_resume(resume_text)      -> ResumeAnalysis
    match_resume_to_jd(jd, resume)   -> MatchReport
    rewrite_resume(jd, resume, match, paragraphs) -> RewriteResult
    score_ats(text, jd)              -> ATSScore
    generate_cover_letter(jd, resume) -> string
    tailor_resume_file(path, jd, want_cover_letter) -> {download_path, ...}
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the package importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from backend.agents import (
    ats_optimizer,
    cover_letter as cover_letter_agent,
    jd_analyzer,
    matcher,
    resume_analyzer,
    rewriter,
)
from backend.agents.orchestrator import run_pipeline
from backend.models import (
    JDAnalysis,
    MatchReport,
    ResumeAnalysis,
    ResumeParagraph,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-resume-match")

server: Server = Server("resume-match-ai")


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="analyze_jd",
            description="Extract structured information from a job description (required skills, keywords, seniority, responsibilities, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "jd_text": {"type": "string", "description": "Raw job description text."}
                },
                "required": ["jd_text"],
            },
        ),
        types.Tool(
            name="analyze_resume",
            description="Extract structured information from raw resume text (candidate name, contact, summary, skills, bullets, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text": {"type": "string", "description": "Raw resume text."}
                },
                "required": ["resume_text"],
            },
        ),
        types.Tool(
            name="match_resume_to_jd",
            description="Compute a gap analysis (covered / weakly covered / missing requirements) between a resume and a JD.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jd": {"type": "object", "description": "JDAnalysis JSON."},
                    "resume": {"type": "object", "description": "ResumeAnalysis JSON."},
                },
                "required": ["jd", "resume"],
            },
        ),
        types.Tool(
            name="score_ats",
            description="Compute an ATS keyword coverage score for arbitrary text against a JD.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "jd": {"type": "object"},
                },
                "required": ["text", "jd"],
            },
        ),
        types.Tool(
            name="generate_cover_letter",
            description="Generate a tailored cover letter from a JD and resume.",
            inputSchema={
                "type": "object",
                "properties": {
                    "jd": {"type": "object"},
                    "resume": {"type": "object"},
                },
                "required": ["jd", "resume"],
            },
        ),
        types.Tool(
            name="tailor_resume_file",
            description="Run the full pipeline on a local .docx (or .pdf) resume file. Returns the path of the tailored .docx along with scores and analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_path": {"type": "string", "description": "Absolute path to the candidate's resume file."},
                    "jd_text": {"type": "string", "description": "Raw job description."},
                    "want_cover_letter": {"type": "boolean", "default": False},
                    "output_dir": {"type": "string", "description": "Optional output directory. Defaults to a temp dir."},
                },
                "required": ["resume_path", "jd_text"],
            },
        ),
    ]


def _ok(payload: Any) -> List[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    try:
        if name == "analyze_jd":
            res = jd_analyzer.analyze(arguments["jd_text"])
            return _ok(res.model_dump())

        if name == "analyze_resume":
            res = resume_analyzer.analyze(arguments["resume_text"])
            return _ok(res.model_dump())

        if name == "match_resume_to_jd":
            jd = JDAnalysis(**arguments["jd"])
            resume = ResumeAnalysis(**arguments["resume"])
            res = matcher.match(jd, resume)
            return _ok(res.model_dump())

        if name == "score_ats":
            jd = JDAnalysis(**arguments["jd"])
            res = ats_optimizer.score(arguments["text"], jd)
            return _ok(res.model_dump())

        if name == "generate_cover_letter":
            jd = JDAnalysis(**arguments["jd"])
            resume = ResumeAnalysis(**arguments["resume"])
            text = cover_letter_agent.generate(jd, resume)
            return [types.TextContent(type="text", text=text)]

        if name == "tailor_resume_file":
            resume_path = arguments["resume_path"]
            jd_text = arguments["jd_text"]
            want_cl = bool(arguments.get("want_cover_letter", False))
            out_dir = Path(arguments.get("output_dir") or tempfile.gettempdir())
            out_dir.mkdir(parents=True, exist_ok=True)
            # Output extension is set by the orchestrator based on input type
            suffix = Path(resume_path).suffix.lower() or ".docx"
            out_path = out_dir / f"tailored-resume{suffix}"
            cover_pdf = out_dir / "cover-letter.pdf" if want_cl else None
            result = run_pipeline(
                resume_docx_path=resume_path,
                jd_text=jd_text,
                output_path=out_path,
                want_cover_letter=want_cl,
                cover_letter_pdf_path=cover_pdf,
            )
            return _ok({
                "output_path": str(result.output_path),
                "output_format": result.output_format,
                "match_score": result.match_report.overall_score,
                "ats_coverage": result.ats_score.keyword_coverage,
                "match_report": result.match_report.model_dump(),
                "ats_score": result.ats_score.model_dump(),
                "cover_letter": result.cover_letter_text,
                "cover_letter_pdf_path": str(result.cover_letter_pdf_path) if result.cover_letter_pdf_path else None,
                "warnings": result.warnings,
            })

        return [types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
    except Exception as e:
        logger.exception("tool %s failed", name)
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
