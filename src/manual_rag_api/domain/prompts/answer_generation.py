"""
Answer-generation prompt templates — domain layer.

Plain strings only — no LLM calls, no infrastructure imports.
Selected by query type in AnswerGenerator (infrastructure/generation).

Five templates match the five query types from the classifier:
  lookup     → exact value / spec
  procedure  → step-by-step how-to
  comparison → multi-model side-by-side
  diagnostic → fault / symptom / corrective action
  general    → catch-all fallback
"""

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise technical assistant for service and maintenance manuals.
Answer questions using ONLY the numbered sources provided.
- If a source contains the answer, cite its number.
- If sources do not fully answer the question, say so explicitly.
- Do not invent facts, part numbers, measurements, or procedures.
- Use exact values from the sources (torque specs, capacities, model numbers).
- NEVER repeat the same fact, value, or step more than once. Each point must be unique.
- Stay strictly on-topic — only include content directly relevant to the specific question asked.
- Do not include generic safety disclaimers unless they are specific to this exact procedure.
"""

# ── Query-type templates ──────────────────────────────────────────────────────

GENERAL_TEMPLATE = """\
QUESTION: {query}

SOURCES:
{context}

Respond with a JSON object — no markdown, no extra text:
{{
  "answer": "<complete answer grounded in the sources above>",
  "citations": [
    {{"source_number": <int>, "reason": "<one sentence: what this source contributed>"}}
  ],
  "missing_info": "<what the sources lack, or empty string if fully answered>"
}}
"""

LOOKUP_TEMPLATE = """\
QUESTION: {query}

SOURCES:
{context}

Instructions: The question asks for a specific value (spec, capacity, torque, pressure, etc.).
- Answer with the exact value from the source. Include units.
- If a DETERMINISTIC TABLE MATCH source is present, use that value — it is exact.
- One or two sentences maximum. Do not elaborate beyond the value and its context.

Respond with a JSON object — no markdown, no extra text:
{{
  "answer": "<exact value with units, e.g. '120 L' or '275 bar'>",
  "citations": [
    {{"source_number": <int>, "reason": "<one sentence: what this source contributed>"}}
  ],
  "missing_info": "<what the sources lack, or empty string if fully answered>"
}}
"""

PROCEDURE_TEMPLATE = """\
QUESTION: {query}

SOURCES:
{context}

Instructions: The question asks how to perform a specific task.
- List ONLY the steps that are directly part of this specific procedure.
- Do NOT include generic safety warnings that apply to all maintenance tasks.
- Do NOT repeat the same step, fact, or value more than once — every step must be unique.
- Include specific warnings, cautions, or torque values only if mentioned for THIS procedure.
- If a source does not directly describe this procedure, do not cite it.
- Keep the answer concise — aim for 5–12 steps maximum.

Respond with a JSON object — no markdown, no extra text:
{{
  "answer": "<numbered step-by-step procedure, 5-12 unique steps max>",
  "citations": [
    {{"source_number": <int>, "reason": "<one sentence: what this source contributed>"}}
  ],
  "missing_info": "<what the sources lack, or empty string if fully answered>"
}}
"""

COMPARISON_TEMPLATE = """\
QUESTION: {query}

SOURCES:
{context}

Instructions: The question asks you to compare two or more models/configurations.
- Identify the subjects being compared (model numbers, configurations, etc.).
- Use a side-by-side format if possible (markdown table or parallel bullet points).
- Only include differences and similarities that are explicitly stated in the sources.
- If a spec is only available for one subject, note it as "not found" for the other.

Respond with a JSON object — no markdown, no extra text:
{{
  "answer": "<comparison in table or structured format>",
  "citations": [
    {{"source_number": <int>, "reason": "<one sentence: what this source contributed>"}}
  ],
  "missing_info": "<what the sources lack, or empty string if fully answered>"
}}
"""

DIAGNOSTIC_TEMPLATE = """\
QUESTION: {query}

SOURCES:
{context}

Instructions: The question describes a fault, symptom, or error code.
- Identify the fault or symptom.
- State the likely cause(s) as listed in the sources.
- Provide the recommended corrective action(s) from the sources.
- If an error/fault code is mentioned, include its official definition.

Respond with a JSON object — no markdown, no extra text:
{{
  "answer": "<fault identification, likely cause, and recommended action>",
  "citations": [
    {{"source_number": <int>, "reason": "<one sentence: what this source contributed>"}}
  ],
  "missing_info": "<what the sources lack, or empty string if fully answered>"
}}
"""

# ── Registry ──────────────────────────────────────────────────────────────────
# Maps query_type → template string.  Used by AnswerGenerator to select
# the right prompt without an if/elif chain.

TEMPLATES: dict[str, str] = {
    "lookup":     LOOKUP_TEMPLATE,
    "procedure":  PROCEDURE_TEMPLATE,
    "comparison": COMPARISON_TEMPLATE,
    "diagnostic": DIAGNOSTIC_TEMPLATE,
    "general":    GENERAL_TEMPLATE,
}
