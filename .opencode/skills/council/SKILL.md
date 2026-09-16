---
name: council
description: Multi-perspective AI council — delegates to the global unified council skill at ~/.config/opencode/skills/council/. Use when user asks to "consult the council", "get multiple perspectives", "debate this", or "stress test this plan".
---

# Council Skill (Project Local)

This skill delegates to the unified global council skill.

## Quick Start

Load the global skill content from `~/.config/opencode/skills/council/SKILL.md` and follow its protocol.

## Modes

- **Quick mode** (default): 4 practical personas (Architect, Skeptic, Pragmatist, Researcher) — fast decisions
- **Deep mode** (`--deep`): Historical figures (Aristotle, Socrates, Ada, Feynman, Torvalds, Sun Tzu) — high-stakes architecture

## Usage

```
@council How should we structure the LangGraph state machine?
@council --deep Should we use ChromaDB or Pinecone for production?
```
