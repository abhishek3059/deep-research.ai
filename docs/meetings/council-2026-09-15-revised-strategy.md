# Council Meeting — Revised Strategy for DeepResearch AI

**Date:** 2026-09-15
**Mode:** Deep (Full Historical Panel)
**Topic:** Revised project strategy for AI Engineer job pivot
**Status:** Concluded — Verdict Delivered

---

## Problem Statement

**User Profile:**
- Name: Abhishek Shukla
- Background: 2+ years Java full-stack experience at Amiti Software Technologies
- Current role: Software Developer - Full Stack & AI Engineering
- Goal: Pivot to AI Engineer role, need to showcase production-grade skills
- Already built: RAG pipelines, MCP server, multi-agent system (FinPilot)

**The Question:**
Should we build the full stack, simplify, or find a middle ground?

---

## Council Verdict

**Option C — Build minimum viable integration of 5 frameworks in 4 weeks.**

### Recommended Plan

1. **Day 1-2:** Add langgraph, rewrite graph.py using StateGraph
2. **Day 3-5:** Add guardrails-ai, implement 3 guards
3. **Day 6-8:** Add crewai, create single CriticAgent
4. **Day 9-11:** Wire ragas_eval.py, add deepeval
5. **Day 12-14:** Write README.md, docs/decisions.md, demo script

### Key Decisions
- LangGraph is highest priority (converts claim to evidence)
- CrewAI should be ONE agent, not four
- Guardrails AI needs 3 guards minimum
- Documentation IS the product
- Win with coherent story, not breadth

---

*Full deliberation: See council task output*
