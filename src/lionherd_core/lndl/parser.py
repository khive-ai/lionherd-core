# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL Parser - Structured Output Parsing (Simplified).

This module provides parsing for LNDL structured output tags only.
The parser transforms token streams from the Lexer into Abstract Syntax Trees.

Design Philosophy:
- Recursive descent for clarity and maintainability
- Structured outputs only (defer semantic operations)
- Clean error messages with line/column context
- Support both namespaced and legacy syntax
- Hybrid approach: Lexer/Parser for structure, regex for content preservation

Grammar (Simplified for Structured Outputs):
    Program    ::= (Lvar | Lact)* OutBlock?
    Lvar       ::= '<lvar' (ID '.' ID ID? | ID) '>' Content '</lvar>'
    Lact       ::= '<lact' (ID '.' ID ID? | ID) '>' FuncCall '</lact>'
    OutBlock   ::= 'OUT{' OutFields '}'
    OutFields  ::= OutField (',' OutField)*
    OutField   ::= ID ':' OutValue
    OutValue   ::= '[' RefList ']' | Literal

Performance:
- Typical LNDL response: <5ms
- Linear complexity O(n) for token stream
- Minimal lookahead (efficient single-pass)
"""

import ast
import re
import warnings
from typing import Any

from .ast import Lact, Lvar, OutBlock, Program, RLvar
from .lexer import Token, TokenType

# Track warned action names to prevent duplicate warnings
_warned_action_names: set[str] = set()

# Python reserved keywords and common builtins
PYTHON_RESERVED = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    "print",
    "input",
    "open",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "type",
}


class ParseError(Exception):
    """Parser error with position information.

    Attributes:
        message: Error description
        token: Token where error occurred
    """

    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(f"Parse error at line {token.line}, column {token.column}: {message}")


class Parser:
    """Recursive descent parser for LNDL structured outputs.

    The parser uses recursive descent to transform token streams from the Lexer
    into Abstract Syntax Trees containing Lvar, Lact, and OutBlock nodes.

    Supports:
    - Namespaced lvars: <lvar Model.field alias>content</lvar>
    - Legacy lvars: <lvar alias>content</lvar>
    - Namespaced lacts: <lact Model.field alias>func(...)</lact>
    - Direct lacts: <lact alias>func(...)</lact>
    - OUT blocks: OUT{field: [refs], field2: value}

    Example:
        >>> from lionherd_core.lndl.lexer import Lexer
        >>> lexer = Lexer("<lvar Report.title t>AI Safety</lvar>\\nOUT{title: [t]}")
        >>> tokens = lexer.tokenize()
        >>> parser = Parser(tokens)
        >>> program = parser.parse()
        >>> len(program.lvars)
        1
        >>> program.out_block.fields["title"]
        ['t']
    """

    def __init__(self, tokens: list[Token], source_text: str | None = None):
        """Initialize parser with token stream.

        Args:
            tokens: Token list from lexer (must include EOF token)
            source_text: Optional source text for raw content extraction
        """
        self.tokens = tokens
        self.pos = 0
        self.source_text = source_text

    def current_token(self) -> Token:
        """Get current token without advancing.

        Returns:
            Current token or EOF if past end
        """
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        """Peek at token ahead without advancing.

        Args:
            offset: Number of tokens to look ahead

        Returns:
            Token at offset or EOF if out of bounds
        """
        peek_pos = self.pos + offset
        if peek_pos >= len(self.tokens):
            return self.tokens[-1]  # EOF token
        return self.tokens[peek_pos]

    def advance(self) -> None:
        """Advance to next token."""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def expect(self, token_type: TokenType) -> Token:
        """Expect specific token type and advance.

        Args:
            token_type: Expected token type

        Returns:
            Matched token

        Raises:
            ParseError: If current token doesn't match expected type
        """
        token = self.current_token()
        if token.type != token_type:
            raise ParseError(f"Expected {token_type.name}, got {token.type.name}", token)
        self.advance()
        return token

    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of given types.

        Args:
            *token_types: Token types to match against

        Returns:
            True if current token matches any type
        """
        return self.current_token().type in token_types

    def skip_newlines(self) -> None:
        """Skip newline tokens."""
        while self.match(TokenType.NEWLINE):
            self.advance()

    def parse(self) -> Program:
        """Parse token stream into AST.

        Uses token-based parsing with regex content extraction to preserve whitespace and quotes.

        Returns:
            Program node containing lvars, lacts, and optional out_block

        Example:
            >>> parser = Parser(tokens, source_text)
            >>> program = parser.parse()
            >>> len(program.lvars)
            2
        """
        if self.source_text is None:
            raise ParseError(
                "Parser requires source_text for content extraction", self.current_token()
            )

        lvars: list[Lvar] = []
        lacts: list[Lact] = []
        out_block: OutBlock | None = None

        # Token-based parsing for all LNDL constructs
        # Track aliases to detect duplicates (lvars and lacts share same namespace)
        aliases: set[str] = set()

        while not self.match(TokenType.EOF):
            self.skip_newlines()

            if self.match(TokenType.EOF):
                break

            # Parse lvar declaration
            if self.match(TokenType.LVAR_OPEN):
                lvar = self.parse_lvar()
                # Check for duplicate alias (lvars and lacts share namespace)
                if lvar.alias in aliases:
                    raise ParseError(
                        f"Duplicate alias '{lvar.alias}' - aliases must be unique across lvars and lacts",
                        self.current_token(),
                    )
                aliases.add(lvar.alias)
                lvars.append(lvar)
                continue

            # Parse lact declaration
            if self.match(TokenType.LACT_OPEN):
                lact = self.parse_lact()
                # Check for duplicate alias (lvars and lacts share namespace)
                if lact.alias in aliases:
                    raise ParseError(
                        f"Duplicate alias '{lact.alias}' - aliases must be unique across lvars and lacts",
                        self.current_token(),
                    )
                aliases.add(lact.alias)
                lacts.append(lact)
                continue

            # Parse OUT{} block
            if self.match(TokenType.OUT_OPEN):
                out_block = self.parse_out_block()
                break

            # Skip all other tokens (narrative text, punctuation, etc.)
            self.advance()

        return Program(lvars=lvars, lacts=lacts, out_block=out_block)

    def parse_lvar(self) -> Lvar | RLvar:
        """Parse lvar declaration (namespaced or raw).

        Grammar:
            Lvar  ::= '<lvar' ID '.' ID ID? '>' Content '</lvar>'  # Namespaced
            RLvar ::= '<lvar' ID '>' Content '</lvar>'              # Raw

        Namespaced pattern (maps to Pydantic model):
            <lvar Model.field alias>content</lvar>
            <lvar Model.field>content</lvar>  # Uses field as alias

        Raw pattern (simple string capture):
            <lvar alias>content</lvar>

        Returns:
            Lvar node (namespaced) or RLvar node (raw)

        Examples:
            <lvar Report.title t>AI Safety Analysis</lvar>
            → Lvar(model="Report", field="title", alias="t", content="AI Safety Analysis")

            <lvar reasoning>The analysis shows...</lvar>
            → RLvar(alias="reasoning", content="The analysis shows...")
        """
        self.expect(TokenType.LVAR_OPEN)  # <lvar
        self.skip_newlines()

        # Parse first identifier
        first_id = self.expect(TokenType.ID).value

        # Check for DOT to distinguish namespaced vs raw pattern
        if self.match(TokenType.DOT):
            # Namespaced pattern: Model.field [alias]
            self.advance()  # consume dot
            field = self.expect(TokenType.ID).value
            model = first_id

            # Check for optional alias (next token before GT)
            if self.match(TokenType.ID):
                alias = self.current_token().value
                self.advance()
                has_explicit_alias = True
            else:
                # No alias - use field name
                alias = field
                has_explicit_alias = False

            is_raw = False

        else:
            # Raw pattern: just alias
            alias = first_id
            model = None
            field = None
            has_explicit_alias = False
            is_raw = True

        # Expect closing '>' for tag
        self.expect(TokenType.GT)
        self.skip_newlines()

        # Extract content directly from source_text to preserve whitespace and quotes
        # Use regex to find the lvar tag and extract content
        if not self.source_text:
            raise ParseError(
                "Parser requires source_text for content extraction", self.current_token()
            )

        # Build regex pattern based on parsed structure
        if is_raw:
            # Raw: <lvar alias>content</lvar>
            pattern = rf"<lvar\s+{re.escape(alias)}\s*>(.*?)</lvar>"
        else:
            # Namespaced
            if has_explicit_alias:
                # Explicit alias: <lvar Model.field alias>content</lvar>
                pattern = rf"<lvar\s+{re.escape(model)}\.{re.escape(field)}\s+{re.escape(alias)}\s*>(.*?)</lvar>"
            else:
                # No alias: <lvar Model.field>content</lvar>
                pattern = rf"<lvar\s+{re.escape(model)}\.{re.escape(field)}\s*>(.*?)</lvar>"

        match = re.search(pattern, self.source_text, re.DOTALL)
        if not match:
            # Check if it's an unclosed tag issue
            if "</lvar>" not in self.source_text:
                raise ParseError("Unclosed lvar tag - missing </lvar>", self.current_token())
            raise ParseError(
                f"Could not extract lvar content with pattern: {pattern}", self.current_token()
            )

        content = match.group(1).strip()

        # Skip tokens until closing tag
        while not self.match(TokenType.LVAR_CLOSE):
            if self.match(TokenType.EOF):
                raise ParseError("Unclosed lvar tag - missing </lvar>", self.current_token())
            self.advance()

        self.expect(TokenType.LVAR_CLOSE)  # </lvar>

        # Return appropriate node type based on pattern
        if is_raw:
            return RLvar(alias=alias, content=content)
        else:
            return Lvar(model=model, field=field, alias=alias, content=content)

    def parse_lact(self) -> Lact:
        """Parse lact (action) declaration.

        Grammar:
            Lact ::= '<lact' (ID '.' ID ID? | ID) '>' FuncCall '</lact>'

        Supports:
            Namespaced: <lact Model.field alias>func(...)</lact>
            Direct: <lact alias>func(...)</lact>

        Returns:
            Lact node with model, field, alias, and call string

        Example:
            <lact Report.summary s>generate_summary(prompt="...")</lact>
            → Lact(model="Report", field="summary", alias="s", call="generate_summary(...)")

            <lact search>search(query="AI")</lact>
            → Lact(model=None, field=None, alias="search", call="search(...)")
        """
        self.expect(TokenType.LACT_OPEN)  # <lact
        self.skip_newlines()

        # Parse identifier (could be Model, field, or alias)
        first_id = self.expect(TokenType.ID).value

        # Track whether alias was explicitly provided
        has_explicit_alias = False

        # Check for namespaced pattern: Model.field [alias]
        if self.match(TokenType.DOT):
            self.advance()  # consume dot
            field = self.expect(TokenType.ID).value
            model = first_id

            # Check for optional alias (next token before GT)
            if self.match(TokenType.ID):
                alias = self.current_token().value
                self.advance()
                has_explicit_alias = True
            else:
                # No alias - use field name
                alias = field
                has_explicit_alias = False

        else:
            # Direct pattern: <lact alias>
            model = None
            field = None
            alias = first_id
            has_explicit_alias = True  # Direct always has explicit alias

        # Expect closing '>' for tag
        self.expect(TokenType.GT)
        self.skip_newlines()

        # Extract call directly from source_text to preserve exact syntax (quotes, spaces, etc.)
        if not self.source_text:
            raise ParseError(
                "Parser requires source_text for call extraction", self.current_token()
            )

        # Build regex pattern based on parsed structure
        if model:
            if has_explicit_alias:
                # Explicit alias: <lact Model.field alias>call</lact>
                pattern = rf"<lact\s+{re.escape(model)}\.{re.escape(field)}\s+{re.escape(alias)}\s*>(.*?)</lact>"
            else:
                # No alias: <lact Model.field>call</lact>
                pattern = rf"<lact\s+{re.escape(model)}\.{re.escape(field)}\s*>(.*?)</lact>"
        else:
            # Direct: <lact alias>call</lact>
            pattern = rf"<lact\s+{re.escape(alias)}\s*>(.*?)</lact>"

        match = re.search(pattern, self.source_text, re.DOTALL)
        if not match:
            # Check if it's an unclosed tag issue
            if "</lact>" not in self.source_text:
                raise ParseError("Unclosed lact tag - missing </lact>", self.current_token())
            raise ParseError(
                f"Could not extract lact call with pattern: {pattern}", self.current_token()
            )

        call = match.group(1).strip()

        # Skip tokens until closing tag
        while not self.match(TokenType.LACT_CLOSE):
            if self.match(TokenType.EOF):
                raise ParseError("Unclosed lact tag - missing </lact>", self.current_token())
            self.advance()

        self.expect(TokenType.LACT_CLOSE)  # </lact>

        # Warn if using reserved keyword as action name
        if alias in PYTHON_RESERVED and alias not in _warned_action_names:
            _warned_action_names.add(alias)
            warnings.warn(
                f"Action name '{alias}' is a Python reserved keyword or builtin.",
                UserWarning,
                stacklevel=2,
            )

        return Lact(model=model, field=field, alias=alias, call=call)

    def parse_out_block(self) -> OutBlock:
        """Parse OUT{} block.

        Grammar:
            OutBlock  ::= 'OUT{' OutFields '}'
            OutFields ::= OutField (',' OutField)*
            OutField  ::= ID ':' OutValue
            OutValue  ::= '[' RefList ']' | Literal
            RefList   ::= ID (',' ID)*

        Returns:
            OutBlock node with fields dict

        Example:
            OUT{title: [t], summary: [s], confidence: 0.85}
            → OutBlock(fields={"title": ["t"], "summary": ["s"], "confidence": 0.85})
        """
        self.expect(TokenType.OUT_OPEN)  # OUT{
        self.skip_newlines()

        fields: dict[str, list[str] | str | int | float | bool] = {}

        # Parse field assignments
        while not self.match(TokenType.OUT_CLOSE, TokenType.EOF):
            self.skip_newlines()

            if self.match(TokenType.OUT_CLOSE, TokenType.EOF):
                break

            # Parse field name
            if not self.match(TokenType.ID):
                # Skip unexpected tokens
                self.advance()
                continue

            field_name = self.current_token().value
            self.advance()

            self.skip_newlines()

            # Expect colon (raises ParseError if not found)
            self.expect(TokenType.COLON)
            self.skip_newlines()

            # Parse value - either [ref1, ref2] or literal
            if self.match(TokenType.LBRACKET):
                # Array of references
                self.advance()  # consume [
                self.skip_newlines()

                refs: list[str] = []
                while not self.match(TokenType.RBRACKET, TokenType.EOF):
                    self.skip_newlines()

                    if self.match(TokenType.RBRACKET, TokenType.EOF):
                        break

                    if self.match(TokenType.ID):
                        refs.append(self.current_token().value)
                        self.advance()
                    else:
                        # Only IDs allowed in arrays (variable/action references)
                        # Literals not supported - resolver expects references only
                        raise ParseError(
                            f"Arrays must contain only variable/action references (IDs), "
                            f"not literals. Got: {self.current_token().type}",
                            self.current_token(),
                        )

                    self.skip_newlines()

                    # Skip comma if present
                    if self.match(TokenType.COMMA):
                        self.advance()

                if self.match(TokenType.RBRACKET):
                    self.advance()  # consume ]

                fields[field_name] = refs

            elif self.match(TokenType.STR):
                # String literal
                fields[field_name] = self.current_token().value
                self.advance()

            elif self.match(TokenType.NUM):
                # Number literal
                num_str = self.current_token().value
                self.advance()

                # Convert to int or float
                if "." in num_str:
                    fields[field_name] = float(num_str)
                else:
                    fields[field_name] = int(num_str)

            elif self.match(TokenType.ID):
                # Could be boolean (true/false) or single reference
                value = self.current_token().value
                self.advance()

                if value.lower() == "true":
                    fields[field_name] = True
                elif value.lower() == "false":
                    fields[field_name] = False
                else:
                    # Treat as single reference - wrap in list
                    fields[field_name] = [value]

            else:
                # Unknown value type - skip
                self.advance()

            self.skip_newlines()

            # Skip optional comma
            if self.match(TokenType.COMMA):
                self.advance()

        if self.match(TokenType.OUT_CLOSE):
            self.advance()  # consume }

        return OutBlock(fields=fields)


def parse_value(value_str: Any) -> Any:
    """Parse string value to Python object."""
    if not isinstance(value_str, str):
        return value_str
    value_str = value_str.strip()
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null":
        return None
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


__all__ = ("ParseError", "Parser")
