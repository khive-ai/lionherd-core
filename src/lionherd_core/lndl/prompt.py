# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

LNDL_SYSTEM_PROMPT = """LNDL - Structured Output with Natural Thinking

SYNTAX

Variables:
<lvar Model.field alias>value</lvar>

- Model.field: Explicit mapping (Report.title, Reason.confidence)
- alias: Short name for OUT{} reference (optional, defaults to field name)
- Declare anywhere, revise anytime, think naturally

Output:
```lndl
OUT{field1:[var1, var2], field2:[var3], scalar:literal}
```

Arrays for models, literals for scalars (float, str, int, bool)

EXAMPLE

Specs: report(Report: title, summary), reasoning(Reason: confidence, analysis), quality_score(float)

Let me think through this...
<lvar Report.title t>Good Title</lvar>

Initial confidence 70%... wait, more evidence: 85%
<lvar Reason.confidence c>0.85</lvar>
<lvar Report.summary s1>First draft summary</lvar>
<lvar Reason.analysis a>Analysis text</lvar>

Let me revise the summary:
<lvar Report.summary s2>Better summary after review</lvar>

```lndl
OUT{report:[t, s2], reasoning:[c, a], quality_score:0.8}
```

KEY POINTS

- Model.field provides explicit mapping (no ambiguity)
- Declare multiple versions (s1, s2), select final in OUT{}
- Think naturally: prose + variables intermixed
- Array syntax: field:[var1, var2] maps to model fields
- Scalar literals: field:0.8 or field:true for simple types
- Unused variables ignored but preserved for debugging

SCALARS vs MODELS

Scalars (float, str, int, bool):
- Can use direct literals: quality:0.8, is_draft:false
- Or single variable: quality:[q]

Models (Pydantic classes):
- Must use array syntax: report:[title, summary]
- Allows revision tracking and field mapping

ERRORS TO AVOID

<lvar title>value</lvar>              # WRONG: Missing Model.field prefix
<lvar Report.title>val</var>          # WRONG: Mismatched tags
OUT{report:Report(title=t)}           # WRONG: No constructors, use arrays
OUT{report:[t, s2], reason:[c, a]}    # WRONG: field name must match spec
OUT{quality_score:[x, y]}             # WRONG: scalars need single var or literal

CORRECT

<lvar Model.field alias>value</lvar>  # Proper namespace
OUT{report:[alias1, alias2]}          # Array maps to model fields
OUT{quality_score:0.8}                # Scalar literal
OUT{quality_score:[q]}                # Scalar from variable
"""


def get_lndl_system_prompt() -> str:
    """Get the LNDL system prompt for LLM guidance.

    Returns:
        LNDL system prompt string
    """
    return LNDL_SYSTEM_PROMPT.strip()
