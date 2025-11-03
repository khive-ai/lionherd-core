# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

LNDL_SYSTEM_PROMPT = """LNDL - Structured Output with Natural Thinking

SYNTAX

Variables:
<lvar Model.field alias>value</lvar>

- Model.field: Explicit mapping (Report.title, Reason.confidence)
- alias: Short name for OUT{} reference (optional, defaults to field name)
- Declare anywhere, revise anytime, think naturally

Actions:
<lact name>function_call(arg1="value", arg2=123)</lact>

- name: Local reference for OUT{} block
- function_call: Pythonic function syntax with arguments
- Only actions referenced in OUT{} are executed
- Actions NOT in OUT{} are scratch work (thinking, not executed)

Output:
```lndl
OUT{field1:[var1, var2], field2:[action], scalar:literal}
```

Arrays for models, action references for tool results, literals for scalars (float, str, int, bool)

EXAMPLE

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
- Can mix lvars and actions: data:[title, api_call, summary]
- Actions referenced are executed, results used as field values

Actions (tool/function calls):
- Declared with <lact name>function(args)</lact>
- Referenced by name in OUT{}: field:[action_name]
- Only executed if in OUT{} (scratch actions ignored)
- Use pythonic call syntax: search(query="text", limit=10)

ERRORS TO AVOID

<lvar title>value</lvar>                    # WRONG: Missing Model.field prefix
<lvar Report.title>val</var>                # WRONG: Mismatched tags
<lact search>search(...)</lvar>             # WRONG: Mismatched tags (should be </lact>)
OUT{report:Report(title=t)}                 # WRONG: No constructors, use arrays
OUT{report:[t, s2], reason:[c, a]}          # WRONG: field name must match spec
OUT{quality_score:[x, y]}                   # WRONG: scalars need single var or literal
<lact data>search(...)</lact>
<lvar Report.field data>value</lvar>
OUT{field:[data]}                           # WRONG: name collision (both lvar and lact)

CORRECT

<lvar Model.field alias>value</lvar>        # Proper namespace for variables
<lact name>function(args)</lact>            # Proper action declaration
OUT{report:[var1, var2]}                    # Array maps to model fields
OUT{data:[action_name]}                     # Action referenced, will execute
OUT{quality_score:0.8}                      # Scalar literal
OUT{quality_score:[q]}                      # Scalar from variable
OUT{result:[action]}                        # Scalar from action result
"""


def get_lndl_system_prompt() -> str:
    """Get the LNDL system prompt for LLM guidance.

    Returns:
        LNDL system prompt string
    """
    return LNDL_SYSTEM_PROMPT.strip()
