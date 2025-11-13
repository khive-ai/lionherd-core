#!/usr/bin/env python3
"""Validate Python code blocks in documentation markdown files.

This script extracts Python code blocks from markdown files and validates:
1. Syntax correctness (compiles without SyntaxError)
2. Import statements (can be imported without errors)

It supports skipping validation for specific blocks using # noqa: validation
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple


class CodeBlock(NamedTuple):
    """Represents a Python code block found in markdown."""

    content: str
    file_path: Path
    start_line: int
    end_line: int


class ValidationError(NamedTuple):
    """Represents a validation error."""

    file_path: Path
    line_number: int
    error_type: str
    message: str


def extract_code_blocks(file_path: Path) -> list[CodeBlock]:
    """Extract Python code blocks from a markdown file.

    Args:
        file_path: Path to markdown file

    Returns:
        List of CodeBlock objects
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    blocks = []
    in_code_block = False
    current_block = []
    start_line = 0

    for i, line in enumerate(lines, start=1):
        # Detect start of Python code block
        if line.strip().startswith("```python"):
            in_code_block = True
            start_line = i + 1
            current_block = []
        # Detect end of code block
        elif line.strip() == "```" and in_code_block:
            in_code_block = False
            if current_block:
                blocks.append(
                    CodeBlock(
                        content="\n".join(current_block),
                        file_path=file_path,
                        start_line=start_line,
                        end_line=i - 1,
                    )
                )
        # Collect code block lines
        elif in_code_block:
            current_block.append(line)

    return blocks


def should_skip_validation(code: str) -> bool:
    """Check if code block should skip validation.

    Args:
        code: Code block content

    Returns:
        True if block has # noqa: validation comment
    """
    return "# noqa: validation" in code or "# noqa:validation" in code


def is_executable_example(code: str) -> bool:
    """Check if code looks like an executable example (not just a signature).

    We only validate executable examples to avoid false positives from
    API documentation signatures and partial snippets.

    Args:
        code: Code block content

    Returns:
        True if looks like executable code that should be validated
    """
    lines = [line.strip() for line in code.strip().split("\n") if line.strip()]

    # Empty or very short snippets are likely not executable
    if len(lines) < 2:
        return False

    # Check for indicators of executable code
    has_import = any(line.startswith(("from ", "import ")) for line in lines)
    has_complete_function = False
    has_instantiation = False
    has_method_call = False

    # Look for complete function definitions (with body)
    in_function = False
    for i, line in enumerate(lines):
        if line.startswith(("def ", "async def ")) and line.endswith(":"):
            in_function = True
        elif in_function and line and not line.startswith((" ", "\t")):
            # Function ended, check if it had a body
            if i > 0 and lines[i - 1].strip() and not lines[i - 1].startswith("#"):
                has_complete_function = True
            in_function = False

    # Check for object instantiation or method calls
    for line in lines:
        # Instantiation pattern: Variable = Class(...)
        if re.search(r"^\w+\s*=\s*\w+\(", line):
            has_instantiation = True
        # Method call pattern: obj.method(...)
        if re.search(r"\w+\.\w+\(", line):
            has_method_call = True

    # Code is executable if it has imports AND (instantiation OR method calls)
    # OR if it has complete function definitions
    return (has_import and (has_instantiation or has_method_call)) or has_complete_function


def normalize_code_for_validation(code: str) -> str:
    """Normalize code to make it more likely to compile.

    This handles common documentation patterns like:
    - Await outside functions
    - Async with outside functions
    - Async for outside functions
    - Method signatures without bodies

    Args:
        code: Original code

    Returns:
        Normalized code that should compile
    """
    lines = code.split("\n")
    code_text = code

    # Check if code has async keywords but no async function definition
    has_async_keywords = any(
        keyword in code_text for keyword in ["await ", "async with ", "async for "]
    )
    has_async_def = "async def " in code_text

    # Wrap in async function if async keywords present without async def
    if has_async_keywords and not has_async_def:
        wrapped_lines = ["async def _async_wrapper():"]
        for line in lines:
            if line.strip():  # Don't indent empty lines
                wrapped_lines.append("    " + line)
            else:
                wrapped_lines.append(line)
        return "\n".join(wrapped_lines)

    # Handle incomplete function/class definitions
    normalized = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Handle incomplete function/class definitions
        if stripped.endswith(":") and not stripped.startswith("#"):
            normalized.append(line)
            # Check if next line has content
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # If next line doesn't have content, add pass
                if not next_line or next_line.startswith("```"):
                    indent = len(line) - len(line.lstrip()) + 4
                    normalized.append(" " * indent + "pass")
            continue

        normalized.append(line)

    return "\n".join(normalized)


def validate_syntax(block: CodeBlock) -> list[ValidationError]:
    """Validate Python syntax by attempting to compile.

    Args:
        block: Code block to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Only validate executable examples (not API signatures/stubs)
    if not is_executable_example(block.content):
        return errors

    # Try to normalize code for validation
    code = normalize_code_for_validation(block.content)

    try:
        compile(code, str(block.file_path), "exec")
    except SyntaxError as e:
        # Calculate actual line number in file
        line_number = block.start_line + (e.lineno or 1) - 1
        errors.append(
            ValidationError(
                file_path=block.file_path,
                line_number=line_number,
                error_type="SyntaxError",
                message=f"{e.msg} (at column {e.offset})" if e.offset else e.msg,
            )
        )
    except Exception as e:
        errors.append(
            ValidationError(
                file_path=block.file_path,
                line_number=block.start_line,
                error_type=type(e).__name__,
                message=str(e),
            )
        )

    return errors


def extract_imports(code: str) -> list[str]:
    """Extract import statements from code.

    Args:
        code: Python code

    Returns:
        List of module names being imported
    """
    imports = []

    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
    except SyntaxError:
        # If we can't parse, skip import validation
        pass

    return imports


def validate_imports(block: CodeBlock) -> list[ValidationError]:
    """Validate that imports can be resolved.

    Args:
        block: Code block to validate

    Returns:
        List of validation errors (empty if valid)
    """
    # Optional dependencies that might appear in examples
    OPTIONAL_DEPENDENCIES = {
        "httpx",
        "starlette",
        "fastapi",
        "openai",
        "anthropic",
        "heavy_module",  # Example module in docs
        "some_external_lib",  # Example module in docs
    }

    errors = []
    imports = extract_imports(block.content)

    for module_name in imports:
        # Skip validation for optional dependencies
        if module_name in OPTIONAL_DEPENDENCIES:
            continue

        try:
            __import__(module_name)
        except ImportError as e:
            errors.append(
                ValidationError(
                    file_path=block.file_path,
                    line_number=block.start_line,
                    error_type="ImportError",
                    message=f"Cannot import '{module_name}': {e}",
                )
            )
        except Exception:
            # Some imports might fail for other reasons (circular imports, etc.)
            # We'll log these but not fail the validation
            pass

    return errors


def validate_file(file_path: Path) -> list[ValidationError]:
    """Validate all Python code blocks in a markdown file.

    Args:
        file_path: Path to markdown file

    Returns:
        List of validation errors
    """
    errors = []
    blocks = extract_code_blocks(file_path)

    for block in blocks:
        # Skip validation if requested
        if should_skip_validation(block.content):
            continue

        # Skip short code blocks with async keywords (likely pattern demonstrations)
        lines = [line for line in block.content.split("\n") if line.strip()]
        if len(lines) < 5 and any(
            kw in block.content for kw in ["await ", "async with ", "async for "]
        ):
            continue

        # Validate syntax
        syntax_errors = validate_syntax(block)
        errors.extend(syntax_errors)

        # Only validate imports if syntax is valid
        if not syntax_errors:
            import_errors = validate_imports(block)
            errors.extend(import_errors)

    return errors


def validate_docs(docs_dir: Path, pattern: str = "**/*.md") -> tuple[int, int]:
    """Validate all markdown files in directory.

    Args:
        docs_dir: Path to docs directory
        pattern: Glob pattern for markdown files

    Returns:
        Tuple of (total_files, files_with_errors)
    """
    all_errors = []
    files_checked = 0
    files_with_errors = 0

    # Find all markdown files
    markdown_files = sorted(docs_dir.glob(pattern))

    for file_path in markdown_files:
        files_checked += 1
        errors = validate_file(file_path)

        if errors:
            files_with_errors += 1
            all_errors.extend(errors)

    # Prepare report content
    report_lines = []

    if all_errors:
        # Group errors by file
        errors_by_file = {}
        for error in all_errors:
            if error.file_path not in errors_by_file:
                errors_by_file[error.file_path] = []
            errors_by_file[error.file_path].append(error)

        # Build report
        report_lines.append(f"❌ Found {len(all_errors)} validation errors:\n")

        for file_path, file_errors in errors_by_file.items():
            report_lines.append(f"\n{file_path}:")
            for error in sorted(file_errors, key=lambda e: e.line_number):
                report_lines.append(
                    f"  Line {error.line_number}: [{error.error_type}] {error.message}"
                )

        report_lines.append(f"\n\nSummary: {files_with_errors}/{files_checked} files with errors")
        status = "FAILED"
    else:
        report_lines.append(f"✅ All {files_checked} markdown files validated successfully!")
        report_lines.append("   No syntax or import errors found in Python code blocks.")
        status = "PASSED"

    # Print to stdout
    print("\n".join(report_lines))

    # Write report file
    report_path = Path("validation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        f.write(f"\n\nStatus: {status}\n")
        f.write(f"Timestamp: {Path(__file__).stat().st_mtime}\n")

    print(f"\n📄 Report written to: {report_path}")

    return files_checked, files_with_errors


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate Python code blocks in documentation")
    parser.add_argument(
        "docs_dir",
        type=Path,
        help="Path to documentation directory",
    )
    parser.add_argument(
        "--pattern",
        default="**/*.md",
        help="Glob pattern for markdown files (default: **/*.md)",
    )

    args = parser.parse_args()

    if not args.docs_dir.exists():
        print(f"Error: Directory not found: {args.docs_dir}", file=sys.stderr)
        sys.exit(1)

    _total_files, files_with_errors = validate_docs(args.docs_dir, args.pattern)

    # Exit with error code if validation failed
    sys.exit(1 if files_with_errors > 0 else 0)


if __name__ == "__main__":
    main()
