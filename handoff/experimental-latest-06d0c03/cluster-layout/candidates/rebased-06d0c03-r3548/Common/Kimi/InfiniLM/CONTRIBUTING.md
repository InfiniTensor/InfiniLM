# Contributing Guide

For development setup, see [Development Guide](#development-guide) below.

## Code

Please review these details before committing, especially for AI-generated code.

### General

1. Keep changes minimal — do not add what is not necessary.
2. Comments are not always better when abundant. Ideally, the code should be self-explanatory.
3. Files must end with a newline.
4. Use Markdown syntax (backtick-fenced) for identifiers in comments and error messages.
5. Comments and error messages must be in English.
6. Comments and error messages should follow the language's conventions first. If the language does not specify, use complete sentences — capitalize the first letter and end with punctuation.

### C++

1. Unless specified in [Adapt New Models](MODELS.md), follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly. Use the default `.clang-format`.
2. Error and warning messages follow the [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages).
3. Initializer list order must match member declaration order.
4. Formatted by `scripts/format.py`

### Python

Unless specified in [Adapt New Models](MODELS.md), follow [PEP 8](https://peps.python.org/pep-0008/) as the primary style guide. For anything PEP 8 does not cover in detail, refer to the [GDScript style guide](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html)—while it targets a different language, its non-syntax conventions are still applicable.

Formatted by `scripts/format.py`

#### Additional Rules

1. **Comments** should be complete English sentences, starting with a capital letter and ending with punctuation. Use Markdown syntax when referencing code within comments.

2. **Docstrings:** Follow [PEP 257](https://peps.python.org/pep-0257/) conventions.

3. Formatted by `scripts/format.py`

## Commits

Commit messages must follow [Conventional Commits](https://www.conventionalcommits.org/).

Existing commit messages may follow the "issue/### - " legacy format with an issue created to describe the case.

## Pull Requests

1. Small PRs should be squashed. Large PRs may keep multiple commits, but each commit must be meaningful and well-formed.
2. PR titles should respect to issue number and case summary.
3. Before merging (or after each stage of changes), build and test on major involved platforms. Include the results in PRs.

## Branches

Branch names use the format `<type>/xxx-yyyy-zzzz`, where `<type>` matches the PR title's Conventional Commits type, and words are joined with hyphens.

Existing branch names may use the legacy format `issue/###`, followed by a suffix when necessary.

---

# Development Guide

Refer to [ReadMe](README.md) and [Adapt New Models](MODELS.md)

## Troubleshooting

1. to be populated
