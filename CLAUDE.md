# AI Essay Detector

Read docs/PROJECT.md first for full context (architecture, signals, dataset plan, evaluation methodology, tech stack).

Build window: 5 focused days, plus buffer for debugging/polish. Before writing any code, produce a day-by-day implementation plan (checked into docs/IMPLEMENTATION_PLAN.md) that breaks the project into daily chunks or individual tasks with a clear "definition of done" per day. Then follow it in order.

Rules:
- Never let the classifier's verdict come from calling an LLM directly — LLMs only produce signals (log-probs, ranks, cross-perplexity). This is a hard constraint from the brief.
- Stop at the end of each day's checkpoint and wait for review before continuing.
- Document dataset provenance in data/sources.md as you build it, not retroactively.