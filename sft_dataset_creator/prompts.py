"""Prompts sent to generation and evaluation models.

Edit this file to change model behavior. Keep instructions about the response
schema and evidence offsets unless the corresponding parsers are changed too.
"""

GENERATOR_SYSTEM_PROMPT = """
You create one high-quality supervised fine-tuning example from the supplied
source excerpt. Return only JSON matching the requested schema.

Requirements:
- Use only facts present in the excerpt.
- Write instruction, input, and output in the requested language.
- Ask directly about the subject and make the final instruction self-contained.
- Never mention or allude to the excerpt, document, text, context, passage, or
  hidden source. Avoid phrases such as "according to the text", "based on the
  document", "cited above", and equivalents in the target language.
- Do not expose chain of thought.
- Evidence offsets are relative to the referenced section.
""".strip()


TASK_INSTRUCTIONS = {
    "definition": "Create a precise definition task grounded only in the excerpt.",
    "closed_qa": "Create a non-trivial closed question with one excerpt-supported answer.",
    "summarization": "Create a focused summarization task preserving the excerpt's important facts.",
    "information_extraction": "Create an information extraction task with an explicit requested structure.",
    "comparison": "Create a comparison task only when the excerpt supports both sides.",
    "classification": "Create a classification task whose label can be justified from the excerpt.",
    "timeline": "Create a chronological task only when multiple dated events are present.",
    "rewrite": "Create a constrained rewrite task without adding facts.",
    "concept_explanation": "Create a didactic explanation task grounded in the excerpt.",
    "structured_extraction": "Create a structured extraction task with a compact output.",
    "fact_checking": "Create a fact-checking task with an answer supported by direct evidence.",
    "taxonomy": "Create a taxonomy task using categories supported by the excerpt.",
    "short_answer": "Create a concise factual question requiring a short answer.",
    "didactic_explanation": "Create a clear teaching-oriented task for the requested difficulty.",
}


JUDGE_SYSTEM_PROMPT = """
Evaluate one SFT candidate using only the supplied evidence. Return JSON only.
Reject unsupported claims, vague instructions, trivial tasks, malformed outputs,
and any candidate that refers to a text, document, context, passage, excerpt, or
other hidden source.
""".strip()


JUDGE_CRITERIA = [
    "factual grounding",
    "self-contained instruction with no references to hidden source material",
    "instruction quality",
    "output quality",
    "non-triviality",
]


DOCTOR_SYSTEM_PROMPT = "Return JSON only."
DOCTOR_USER_PROMPT = 'Return {"ok": true}.'
