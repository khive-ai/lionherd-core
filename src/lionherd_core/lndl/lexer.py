# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL Lexer - Tokenization for Structured Outputs.

This module provides tokenization for LNDL structured output tags.
The lexer transforms LNDL responses into token streams for parsing.

Design Philosophy:
- Clean tokenization with line/column tracking
- Support for structured output tags (lvars, lacts, OUT blocks)
- Robust error reporting with position information
- Minimal lookahead (efficient single-pass)

Token Types:
- Tags: LVAR_OPEN, LVAR_CLOSE, LACT_OPEN, LACT_CLOSE, OUT_OPEN, OUT_CLOSE
- Literals: ID, NUM, STR
- Punctuation: DOT, COMMA, COLON, LBRACKET, RBRACKET, LPAREN, RPAREN
- Control: NEWLINE, EOF
"""

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for LNDL structured outputs."""

    # Tags
    LVAR_OPEN = auto()  # <lvar
    LVAR_CLOSE = auto()  # </lvar>
    LACT_OPEN = auto()  # <lact
    LACT_CLOSE = auto()  # </lact>
    OUT_OPEN = auto()  # OUT{
    OUT_CLOSE = auto()  # }

    # Literals
    ID = auto()  # identifiers
    NUM = auto()  # numbers (int or float)
    STR = auto()  # strings (quoted)

    # Punctuation
    DOT = auto()  # .
    COMMA = auto()  # ,
    COLON = auto()  # :
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    GT = auto()  # >

    # Control
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    """Token with type, value, and position information.

    Attributes:
        type: Token type classification
        value: Literal token value from source
        line: Line number (1-indexed)
        column: Column number (1-indexed)
    """

    type: TokenType
    value: str
    line: int
    column: int


class Lexer:
    """LNDL lexer for structured output tokenization.

    The lexer performs single-pass tokenization with minimal lookahead,
    tracking line/column positions for error reporting.

    Supported Tags:
    - <lvar Model.field alias>content</lvar>
    - <lact Model.field alias>call()</lact>
    - OUT{field: [refs], field2: value}

    Example:
        >>> lexer = Lexer("<lvar Report.title t>AI Safety</lvar>")
        >>> tokens = lexer.tokenize()
        >>> [t.type for t in tokens]
        [TokenType.LVAR_OPEN, TokenType.ID, ...]
    """

    def __init__(self, text: str):
        """Initialize lexer with source text.

        Args:
            text: LNDL response text to tokenize
        """
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1  # 1-indexed column tracking
        self.tokens: list[Token] = []

    def current_char(self) -> str | None:
        """Get current character without advancing.

        Returns:
            Current character or None if at end of input
        """
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def peek_char(self, offset: int = 1) -> str | None:
        """Peek at character ahead without advancing.

        Args:
            offset: Number of characters to look ahead

        Returns:
            Character at offset or None if out of bounds
        """
        peek_pos = self.pos + offset
        if peek_pos >= len(self.text):
            return None
        return self.text[peek_pos]

    def advance(self) -> None:
        """Advance to next character, tracking line/column."""
        if self.pos < len(self.text) and self.text[self.pos] == "\n":
            self.line += 1
            self.column = 0
        else:
            self.column += 1
        self.pos += 1

    def skip_whitespace(self) -> None:
        """Skip whitespace characters except newlines."""
        while (char := self.current_char()) and char in " \t\r":
            self.advance()

    def read_identifier(self) -> str:
        """Read identifier.

        Returns:
            Identifier string (alphanumeric + underscore)
        """
        result = ""
        while (char := self.current_char()) and (char.isalnum() or char == "_"):
            result += char
            self.advance()
        return result

    def read_number(self) -> str:
        """Read numeric literal (integer or float).

        Returns:
            Number string (digits + optional decimal point)
        """
        result = ""
        while (char := self.current_char()) and (char.isdigit() or char == "."):
            result += char
            self.advance()
        return result

    def read_string(self) -> str:
        """Read quoted string with escape sequence handling.

        Supports escape sequences:
        - \\n: newline
        - \\t: tab
        - \\r: carriage return
        - \\\\: backslash
        - \\": double quote
        - \\': single quote

        Returns:
            String content without quotes, with escapes processed
        """
        quote_char = self.current_char()
        self.advance()  # Skip opening quote

        result = ""
        while (char := self.current_char()) and char != quote_char:
            if char == "\\":
                self.advance()
                if escape_char := self.current_char():
                    # Handle escape sequences
                    if escape_char == "n":
                        result += "\n"
                    elif escape_char == "t":
                        result += "\t"
                    elif escape_char == "r":
                        result += "\r"
                    elif escape_char == "\\":
                        result += "\\"
                    elif escape_char == '"':
                        result += '"'
                    elif escape_char == "'":
                        result += "'"
                    else:
                        # Unknown escape: keep literal
                        result += escape_char
                    self.advance()
            else:
                result += char
                self.advance()

        if self.current_char() == quote_char:
            self.advance()  # Skip closing quote

        return result

    def tokenize(self) -> list[Token]:
        """Tokenize LNDL source code into token stream.

        Returns:
            List of tokens including EOF token

        Example:
            >>> lexer = Lexer("OUT{title: [t]}")
            >>> tokens = lexer.tokenize()
            >>> len(tokens)
            8  # OUT_OPEN, ID, COLON, LBRACKET, ID, RBRACKET, OUT_CLOSE, EOF
        """
        in_out_block = False  # Track whether we're inside OUT{} block
        while char := self.current_char():
            # Skip whitespace (not newlines)
            if char in " \t\r":
                self.skip_whitespace()
                continue

            # Newlines
            if char == "\n":
                self.tokens.append(Token(TokenType.NEWLINE, "\n", self.line, self.column))
                self.advance()
                continue

            # Tag opening: <lvar or <lact
            if char == "<":
                start_line = self.line
                start_column = self.column

                # Check for </lvar> closing tag
                if self.text[self.pos : self.pos + 7] == "</lvar>":
                    self.tokens.append(
                        Token(TokenType.LVAR_CLOSE, "</lvar>", start_line, start_column)
                    )
                    self.pos += 7
                    self.column += 7
                    continue

                # Check for </lact> closing tag
                if self.text[self.pos : self.pos + 7] == "</lact>":
                    self.tokens.append(
                        Token(TokenType.LACT_CLOSE, "</lact>", start_line, start_column)
                    )
                    self.pos += 7
                    self.column += 7
                    continue

                # Check for <lvar opening tag
                if self.text[self.pos : self.pos + 5] == "<lvar":
                    self.tokens.append(
                        Token(TokenType.LVAR_OPEN, "<lvar", start_line, start_column)
                    )
                    self.pos += 5
                    self.column += 5
                    continue

                # Check for <lact opening tag
                if self.text[self.pos : self.pos + 5] == "<lact":
                    self.tokens.append(
                        Token(TokenType.LACT_OPEN, "<lact", start_line, start_column)
                    )
                    self.pos += 5
                    self.column += 5
                    continue

                # Not a recognized tag, skip
                self.advance()
                continue

            # OUT{ block opening
            if self.text[self.pos : self.pos + 4] == "OUT{":
                self.tokens.append(Token(TokenType.OUT_OPEN, "OUT{", self.line, self.column))
                self.pos += 4
                self.column += 4
                in_out_block = True  # Now inside OUT{} block
                continue

            # Identifiers
            if char.isalpha() or char == "_":
                start_line = self.line
                start_column = self.column
                identifier = self.read_identifier()
                self.tokens.append(Token(TokenType.ID, identifier, start_line, start_column))
                continue

            # Negative numbers (only inside OUT{} blocks)
            if char == "-" and in_out_block:
                next_char = self.peek_char()
                if next_char and next_char.isdigit():
                    start_line = self.line
                    start_column = self.column
                    self.advance()  # consume minus
                    number = "-" + self.read_number()
                    self.tokens.append(Token(TokenType.NUM, number, start_line, start_column))
                    continue

            # Numbers
            if char.isdigit():
                start_line = self.line
                start_column = self.column
                number = self.read_number()
                self.tokens.append(Token(TokenType.NUM, number, start_line, start_column))
                continue

            # Strings (only tokenize inside OUT{} blocks to avoid narrative text)
            if char in "\"'" and in_out_block:
                start_line = self.line
                start_column = self.column
                string_val = self.read_string()
                self.tokens.append(Token(TokenType.STR, string_val, start_line, start_column))
                continue

            # Single-character punctuation
            if char == ".":
                self.tokens.append(Token(TokenType.DOT, char, self.line, self.column))
            elif char == ",":
                self.tokens.append(Token(TokenType.COMMA, char, self.line, self.column))
            elif char == ":":
                self.tokens.append(Token(TokenType.COLON, char, self.line, self.column))
            elif char == "[":
                self.tokens.append(Token(TokenType.LBRACKET, char, self.line, self.column))
            elif char == "]":
                self.tokens.append(Token(TokenType.RBRACKET, char, self.line, self.column))
            elif char == "(":
                self.tokens.append(Token(TokenType.LPAREN, char, self.line, self.column))
            elif char == ")":
                self.tokens.append(Token(TokenType.RPAREN, char, self.line, self.column))
            elif char == "}":
                self.tokens.append(Token(TokenType.OUT_CLOSE, char, self.line, self.column))
                in_out_block = False  # Exiting OUT{} block
            elif char == ">":
                self.tokens.append(Token(TokenType.GT, char, self.line, self.column))
            else:
                # Unknown character - skip silently
                pass

            self.advance()

        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens


__all__ = ("Lexer", "Token", "TokenType")
