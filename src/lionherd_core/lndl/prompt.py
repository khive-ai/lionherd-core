# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

LNDL_SYSTEM_PROMPT = """LNDL - Structured Output with Natural Thinking

SYNTAX

Variables (two patterns):

Namespaced (for Pydantic models):
<lvar Model.field alias>value</lvar>

- Model.field: Explicit mapping (Report.title, Reason.confidence)
- alias: Short name for OUT{} reference (optional, defaults to field name)
- Use for structured outputs that need type validation

Raw (for simple capture):
<lvar alias>value</lvar>

- alias: Local name for OUT{} reference
- Use for intermediate values or scalar fields
- No model mapping, simple string capture

Both patterns:
- Declare anywhere, revise anytime, think naturally

Actions (two patterns):

Namespaced (for mixing with lvars):
<lact Model.field alias>function_call(args)</lact>

- Model.field: Explicit mapping (like lvars)
- alias: Short name for OUT{} reference (optional, defaults to field name)
- Enables mixing with lvars in same model

Direct (entire output):
<lact name>function_call(args)</lact>

- name: Local reference for OUT{} block
- Result becomes the entire field value
- Cannot mix with lvars (use namespaced pattern instead)

Both patterns:
- Only executed if referenced in OUT{}
- Not in OUT{} = scratch work (thinking, not executed)
- Pythonic function syntax with arguments

Output:
```lndl
OUT{field1:[var1, var2], field2:[action], scalar:literal}
```

Arrays for models, action references for tool results, literals for scalars (float, str, int, bool)

EXAMPLE 1: Direct Actions (entire output)

Specs: report(Report: title, summary), search_data(SearchResults: items, count), quality_score(float)

Let me search first...
<lact broad>search(query="AI", limit=100)</lact>
Too much noise. Let me refine:
<lact focused>search(query="AI safety", limit=20)</lact>

Now I'll analyze the results:
<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.summary s>Based on search results...</lvar>

```lndl
OUT{report:[t, s], search_data:[focused], quality_score:0.85}
```

Note: Only "focused" action executes (in OUT{}). "broad" was scratch work.

EXAMPLE 2: Mixing lvars and namespaced actions

Specs: report(Report: title, summary, footer)

<lvar Report.title t>Analysis Report</lvar>
<lact Report.summary summarize>generate_summary(data="metrics")</lact>
<lvar Report.footer f>End of Report</lvar>

```lndl
OUT{report:[t, summarize, f]}
```

Note: "summarize" action result fills Report.summary field, mixed with lvars for title and footer.

KEY POINTS

- Model.field provides explicit mapping (no ambiguity)
- Declare multiple versions (s1, s2), select final in OUT{}
- Think naturally: prose + variables intermixed
- Array syntax: field:[var1, var2] maps to model fields
- Scalar literals: field:0.8 or field:true for simple types
- Unused variables ignored but preserved for debugging

SCALARS vs MODELS vs ACTIONS

Scalars (float, str, int, bool):
- Can use direct literals: quality:0.8, is_draft:false
- Or single variable: quality:[q]
- Or single action: score:[calculate]

Models (Pydantic classes):
- Must use array syntax: report:[title, summary]
- Can mix lvars and namespaced actions: data:[title, api_call, summary]
- Direct action for entire model: data:[fetch_data] (single action, no lvars)
- Actions referenced are executed, results used as field values

Actions (tool/function calls):
- Namespaced: <lact Model.field name>function(args)</lact> (for mixing)
- Direct: <lact name>function(args)</lact> (entire output)
- Referenced by name in OUT{}: field:[action_name]
- Only executed if in OUT{} (scratch actions ignored)
- Use pythonic call syntax: search(query="text", limit=10)

ERRORS TO AVOID

<lvar Report.title>val</var>                # WRONG: Mismatched tags (should be </lvar>)
<lact search>search(...)</lvar>             # WRONG: Mismatched tags (should be </lact>)
OUT{report:Report(title=t)}                 # WRONG: No constructors, use arrays
OUT{report:[t, s2], reason:[c, a]}          # WRONG: field name must match spec
OUT{quality_score:[x, y]}                   # WRONG: scalars need single var or literal
<lvar x>value</lvar>
OUT{report:[x]}                             # WRONG: raw lvar cannot be used in BaseModel fields
<lact Report.field data>search(...)</lact>
<lvar Report.field data>value</lvar>
OUT{field:[data]}                           # WRONG: name collision (both lvar and lact named "data")

CORRECT

<lvar Model.field alias>value</lvar>              # Namespaced lvar for Pydantic models
<lvar x>value</lvar>                              # Raw lvar for simple capture (scalars only)
<lact Model.field alias>function(args)</lact>     # Namespaced action (for mixing)
<lact name>function(args)</lact>                  # Direct action (entire output)
OUT{report:[var1, var2]}                          # Array maps to model fields (namespaced lvars)
OUT{report:[var1, action1, var2]}                 # Mixing lvars and namespaced actions
OUT{data:[action_name]}                           # Direct action for entire field
OUT{quality_score:0.8}                            # Scalar literal
OUT{quality_score:[q]}                            # Scalar from namespaced or raw lvar
OUT{result:[action]}                              # Scalar from action result
"""


def get_lndl_system_prompt() -> str:
    """Get the LNDL system prompt for LLM guidance.

    Returns:
        LNDL system prompt string
    """
    return LNDL_SYSTEM_PROMPT.strip()
