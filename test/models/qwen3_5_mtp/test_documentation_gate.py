#!/usr/bin/env python3
"""Documentation gate: the description registry and the model guide agree.

This file is the one place where a test reads `MODELS.md` as its input, which
makes it the test a contributor has to update when the guide is restructured:
the anchors it splits on, the section it reads the family table from and the
table's column order are all named here, in `GUIDE_ANCHORS` and
`FAMILY_TABLE_COLUMNS`, instead of being spread through the suite. Everything
else about the guide's wording is free to change.

The guide's family table is what an integrator reads before touching the code,
so a family that exists in only one of the two places is a defect. The guide's
prose is not asserted: only the table rows and the field names the description
reference documents.
"""

import ast
import io
import os
import re
import shutil
import sys
import tempfile
import tokenize
import unittest
from dataclasses import fields

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_TEST_DIR, "..", "llama"))
sys.path.insert(0, _TEST_DIR)

from infinilm.draft_spec import (  # noqa: E402
    DraftModelSpec,
    DraftWeightMap,
    list_draft_model_specs,
)

MODELS_PATH = os.path.abspath(os.path.join(_TEST_DIR, "..", "..", "..", "MODELS.md"))
# The token kind that carries an f-string's literal text, where the running
# interpreter has one. Before PEP 701 (Python 3.12) an f-string was a single
# `STRING` token, so the name does not exist there: probing for it and falling
# back to `None` keeps the reader working on both, because the `STRING` branch
# already covers the older layout. See `code_lines`.
FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
FSTRING_START = getattr(tokenize, "FSTRING_START", None)
FSTRING_END = getattr(tokenize, "FSTRING_END", None)
# A replacement field inside an f-string: a value, not text a reader sees. Used
# by `literal_text` for the layout where an f-string arrives whole.
FORM_FIELD = re.compile(r"\{[^{}]*\}")


def _tokens(text):
    """The token stream `code_lines` reads.

    A module-level indirection so a test can model the tokenizer of an
    interpreter that is not the one running: before Python 3.12 an f-string is
    one `STRING` token, which this interpreter's tokenizer never produces and
    therefore cannot demonstrate. Everything else in the guard stays the real
    code path.
    """
    return tokenize.generate_tokens(io.StringIO(text).readline)


def current_scan_stream():
    """A scan function for the layout this interpreter produces.

    `code_lines` reading this interpreter's tokens, or `code_lines` reading the
    modelled older layout when a test has swapped the tokenizer out. Tests that
    only care that an f-string is read call whatever this returns, so they hold
    on either layout without asserting anything about the token kinds.
    """
    if FSTRING_START is None or FSTRING_END is None:
        return lambda text: code_lines(text, token_source=tokens_without_pep701)
    return code_lines


def tokens_without_pep701(text):
    """The stream a pre-3.12 tokenizer produces: an f-string as one `STRING`.

    Every `FSTRING_START` … `FSTRING_END` span is collapsed into a single
    `STRING` token whose text is that span as written. The span's text is taken
    from the source lines rather than from the pieces, so the token covers the
    source exactly as the older tokenizer reported it. Where the running
    tokenizer has no such span (it is already the older layout) the stream is
    returned unchanged, which is the same thing.
    """
    if FSTRING_START is None or FSTRING_END is None:
        return tokenize.generate_tokens(io.StringIO(text).readline)
    lines = text.split("\n")
    raw = list(tokenize.generate_tokens(io.StringIO(text).readline))
    out, index = [], 0
    while index < len(raw):
        token = raw[index]
        if token.type == FSTRING_START:
            start = token.start
            scan = index + 1
            while raw[scan].type != FSTRING_END:
                scan += 1
            end = raw[scan].end
            whole = "\n".join(lines[row - 1] for row in range(start[0], end[0] + 1))
            out.append(
                tokenize.TokenInfo(tokenize.STRING, whole, start, end, token.line)
            )
            index = scan + 1
            continue
        out.append(token)
        index += 1
    return iter(out)


MODELS_ZH_PATH = os.path.abspath(
    os.path.join(_TEST_DIR, "..", "..", "..", "MODELS_ZH.md")
)
PACKAGE_DIR = os.path.abspath(
    os.path.join(_TEST_DIR, "..", "..", "..", "python", "infinilm")
)
# The guide defines each criterion as a bold cell of its **criteria table**, so
# "defined" is read from that table and not from any other table in the file.
CRITERION_DEFINITION = re.compile(r"^\*\*(C\d+)\*\*$")
# Criterion table header, used to pick the table inside the criteria section.
CRITERION_TABLE_HEADER = "Criterion"
# A reference in the code names a criterion either as `C<n>` (with an optional
# suffix, e.g. the `C6b` this gate was written for) or as an item of the guide's
# criteria list. A `criterion` mention followed by anything else is reported as
# unresolved rather than skipped, so a new phrasing cannot slip past silently.
CRITERION_MENTION = re.compile(r"\bcriteri(?:on|a)\b", re.IGNORECASE)
CRITERION_NAMED = re.compile(r"C(\d+)([a-z]?)")
CRITERION_ITEM = re.compile(r"(?:list\s+)?items?\s+(\d+)", re.IGNORECASE)
# What may sit between `criterion` and its identifier: nothing, or a separator
# a writer reaches for when the number is called out (`criterion: 7`,
# `criterion (C7)`, `criterion #7`). Those forms read as references and are
# parsed rather than skipped.
CRITERION_REFERENCE_PREFIX = re.compile(r"^[\s:#\-\u2013\u2014]*(?:\(\s*)?")
# A mention is a *reference* when the text after it looks like an identifier.
# Prose ("the criteria a description has to satisfy") is not a reference and is
# skipped; an identifier-looking tail this parser cannot read is reported. A
# bare number counts as identifier-looking: `criterion 7` is a reader pointing
# at a criterion, and a form this parser cannot place has to be visible.
CRITERION_LIKE = re.compile(r"(?:C\d|\d|lists?\b|items?\b)", re.IGNORECASE)
# Section headings this gate locates the family table, the field reference and
# the criteria table by. Renaming a heading in `MODELS.md` means updating the
# anchor here, on purpose: the failure says which anchor stopped matching.
GUIDE_ANCHORS = {
    "family_table": "### 6.5 Status of the surveyed families",
    "family_table_end": "\n---\n",
    "description_reference": "### 6.2 The description",
    "description_reference_end": "### 6.3",
    "criteria": "### 6.1 Does my checkpoint qualify? (criteria)",
    "criteria_end": "### 6.2 The description",
}
# Columns the family table is expected to carry, by meaning. The table's header
# names them and the order they appear in is read from the header, so inserting
# a column does not break the gate.
FAMILY_TABLE_COLUMNS = {
    "family": "Registered description",
    "status": "Status here",
}
# What a row without a registered description has to say about itself.
UNDESCRIBED_MARKER = "not described yet"
# The claim the translated guide has to make, and the words that turn it into
# the opposite claim.
AUTHORITY_PHRASE = "为权威"
# Text that may sit between the file name and the phrase because it only names
# the file again: `MODELS.md`（英文版）为权威版本.
AUTHORITY_ANNOTATIONS = ("（英文版）", "(英文版)", "`", "**", " ", "\n", ">")
NEGATION_WORDS = ("并非", "不是", "不为", "非", "不")


class GuideAnchorError(AssertionError):
    """A heading this gate locates its input by is not in the guide."""


def load_guide():
    with open(MODELS_PATH, encoding="utf-8") as f:
        return f.read()


def slice_section(guide, anchor, end_anchor):
    """Return the text between two headings, failing with the anchor's name."""
    start = guide.find(anchor)
    if start < 0:
        raise GuideAnchorError(
            f"the guide has no {anchor!r} heading; this gate reads the family"
            " table from that section"
        )
    end = guide.find(end_anchor, start + len(anchor))
    if end < 0:
        raise GuideAnchorError(
            f"the guide has no {end_anchor!r} heading after {anchor!r}; this"
            " gate needs the section to end somewhere"
        )
    return guide[start:end]


def split_row(line):
    """Split one markdown table row into cells; None for a separator row."""
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if all(set(cell) <= {"-", ":"} and cell for cell in cells):
        return None
    return cells


def table_rows(text):
    """Rows of the first markdown table in `text`, as lists of cells."""
    lines = [line for line in text.splitlines() if line.lstrip().startswith("|")]
    rows = []
    for line in lines:
        cells = split_row(line)
        if cells is not None:
            rows.append(cells)
    return rows


def family_table(guide):
    """The family table's rows and its column order, read from the section."""
    section = slice_section(
        guide, GUIDE_ANCHORS["family_table"], GUIDE_ANCHORS["family_table_end"]
    )
    rows = table_rows(section)
    if not rows:
        raise GuideAnchorError(
            f"no table found under {GUIDE_ANCHORS['family_table']!r}; this gate"
            " reads the family table from there"
        )
    header = rows[0]
    columns = {}
    for meaning, title in FAMILY_TABLE_COLUMNS.items():
        if title not in header:
            raise GuideAnchorError(
                f"the family table has no {title!r} column; this gate reads the"
                f" {meaning!r} from it"
            )
        columns[meaning] = header.index(title)
    return rows[1:], columns


def described_families():
    return {spec.family for spec in list_draft_model_specs()}


class RegistryDocumentationTest(unittest.TestCase):
    """The description registry and the model guide must agree."""

    @classmethod
    def setUpClass(cls):
        cls.guide = load_guide()

    def rows(self):
        rows, columns = family_table(self.guide)
        return rows, columns

    def test_every_described_family_is_in_the_family_table(self):
        rows, columns = self.rows()
        listed = {
            row[columns["family"]].strip("`")
            for row in rows
            if row[columns["family"]].startswith("`")
        }
        self.assertEqual(described_families() - listed, set())

    def test_the_table_states_the_same_availability_as_the_registry(self):
        rows, columns = self.rows()
        by_family = {
            row[columns["family"]].strip("`"): row
            for row in rows
            if row[columns["family"]].startswith("`")
        }
        for spec in list_draft_model_specs():
            row = by_family[spec.family]
            row_text = " | ".join(row)
            for target in spec.target_model_types:
                self.assertIn(target, row_text)
            status = row[columns["status"]]
            if spec.is_available:
                self.assertIn("block implemented here", status)
            else:
                self.assertIn("block not implemented here", status)
                # The primary missing piece is quoted from the registry.
                self.assertIn(spec.unimplemented[0], status)

    def test_families_the_guide_names_are_registered(self):
        rows, columns = self.rows()
        described = described_families()
        for row in rows:
            if row[columns["family"]].startswith("`"):
                self.assertIn(row[columns["family"]].strip("`"), described)

    def test_surveyed_families_without_a_description_say_so(self):
        # Rows the guide surveyed but this build has not described carry a dash
        # in the registry column and must say the family is not described yet.
        rows, columns = self.rows()
        for row in rows:
            if not row[columns["family"]].startswith("`"):
                self.assertIn(UNDESCRIBED_MARKER, row[columns["status"]])

    def description_reference(self):
        return slice_section(
            self.guide,
            GUIDE_ANCHORS["description_reference"],
            GUIDE_ANCHORS["description_reference_end"],
        )

    def field_column(self):
        """The combined text of the field table's Fields column."""
        rows = table_rows(self.description_reference())
        header = next(row for row in rows if "Fields" in row)
        index = header.index("Fields")
        return " ".join(row[index] for row in rows if len(row) > index)

    def test_every_description_field_is_documented(self):
        # The field reference is what an integrator fills in, so a field that
        # exists in code but not in the guide is a defect in either of them.
        # It has to be in the table's **Fields** column: a mention in a
        # neighbouring cell does not tell an integrator what to fill in.
        field_column = self.field_column()
        for dataclass in (DraftModelSpec, DraftWeightMap):
            for field in fields(dataclass):
                self.assertIn(f"`{field.name}`", field_column)

    def test_the_field_column_check_rejects_a_field_moved_out_of_it(self):
        # Control: take a field out of the Fields column but leave its name in
        # the Meaning column of the same row, which is what the previous
        # "appears anywhere in the section" check accepted.
        rows = table_rows(self.description_reference())
        header = next(row for row in rows if "Fields" in row)
        index = header.index("Fields")
        moved = []
        for row in rows:
            if len(row) > index and "`target_model_types`" in row[index]:
                row = list(row)
                row[index] = row[index].replace("`target_model_types`, ", "")
                row[index + 1] += " (`target_model_types`)"
            moved.append(row)
        header = next(row for row in moved if "Fields" in row)
        index = header.index("Fields")
        field_column = " ".join(row[index] for row in moved if len(row) > index)
        self.assertNotIn("`target_model_types`", field_column)


def defined_criteria(guide):
    """Criterion identifiers the guide's **criteria table** defines.

    Only the criteria section is read, and only its table: a bold `C<n>` cell
    in some other table of the guide is not a definition, so adding one cannot
    legitimise a reference.
    """
    section = slice_section(
        guide, GUIDE_ANCHORS["criteria"], GUIDE_ANCHORS["criteria_end"]
    )
    tables = []
    for line in section.splitlines():
        if line.lstrip().startswith("|"):
            cells = split_row(line)
            if cells is not None:
                tables.append(cells)
    defined = set()
    for row in tables:
        for cell in row:
            match = CRITERION_DEFINITION.match(cell)
            if match:
                defined.add(match.group(1))
    return defined


def literal_text(token_string):
    """The text a string literal holds, without its quotes or escapes.

    A message split across source lines ends up with a literal `\\n` in the
    middle of the pieces, which would hide the word after it; the escapes are
    decoded so the scan reads the text a reader sees.

    An f-string literal reaches this function as a whole `STRING` token only on
    the layout where the tokenizer does not split it (before Python 3.12); its
    literal text is the body with each `{…}` field emptied, since a field is a
    value rather than text a reader sees.
    """
    body = token_string.strip()
    for prefix in ("rb", "br", "R", "B", "r", "b", "F", "f", "u", "U"):
        if body.startswith(prefix):
            body = body[len(prefix) :]
            break
    for quote in ('"""', "'''", '"', "'"):
        if (
            body.startswith(quote)
            and body.endswith(quote)
            and len(body) >= 2 * len(quote)
        ):
            body = body[len(quote) : -len(quote)]
            break
    # The text a reader sees: line and quote escapes decoded, no trimming,
    # because whether a piece ends in a space is part of what is read. Only
    # the escapes a string body actually carries are decoded, so a regex
    # fragment inside the string (`\d`) is left alone instead of turning
    # into a warning and a mangled pattern.
    replacements = (
        ("\\n", "\n"),
        ("\\r", "\r"),
        ("\\t", "\t"),
        ('\\"', '"'),
        ("\\'", "'"),
        ("\\\\", "\\"),
    )
    for source, target in replacements:
        body = body.replace(source, target)
    return FORM_FIELD.sub("", body)


def code_lines(text, token_source=None):
    """The strings a reader would see, as one entry per source line.

    A `criterion` mention that matters is always inside a string — an error
    message is a string, and so is the quote a message is built from — while a
    docstring and a comment are prose that must not be scanned. Both facts are
    needed, so both are used: the tree says where the docstrings are (which the
    token stream alone cannot tell from any other string), and the token spans
    say where every string begins and ends (which the tree alone cannot tell for
    an implicit concatenation).

    A literal's text does not always arrive as a `STRING` token. From Python 3.12
    (PEP 701) an f-string is tokenized as `FSTRING_START`, one or more
    `FSTRING_MIDDLE` pieces (the text a reader sees, on either side of each
    `{…}`) and `FSTRING_END`, so a message built with an f-string carries its
    reference in a `FSTRING_MIDDLE` and nothing else would see it. Before 3.12
    the same f-string is one `STRING` token, which the first branch already
    covers — so the second branch is guarded by a capability probe rather than a
    version check: where the token kind does not exist there is nothing to read
    out of it, and nothing is lost.

    Line numbers are preserved by emitting one entry per source line, blank for
    the lines nothing survived on, so a failure still points at the line the
    text came from.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {1: [text]}
    try:
        tokens = list((token_source or _tokens)(text))
    except (tokenize.TokenError, IndentationError):
        return {1: [text]}

    def span(node):
        return node.lineno, node.end_lineno or node.lineno

    # Docstrings, whole: a bare string expression that is the first statement of
    # a module, class or function.
    prose = []
    for node in ast.walk(tree):
        for field in ("body",):
            body = getattr(node, field, None)
            if not isinstance(body, list) or not body:
                continue
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                prose.append(span(first.value))

    kept = {}
    for token in tokens:
        if token.type == tokenize.STRING:
            text_of = literal_text(token.string)
        elif FSTRING_MIDDLE is not None and token.type == FSTRING_MIDDLE:
            # The literal text of an f-string piece, already unquoted; its
            # escapes are decoded the way a plain literal's are.
            text_of = literal_text('"' + token.string + '"')
        else:
            continue
        first, last = token.start[0], token.end[0]
        if any(start_row <= first and last <= end_row for start_row, end_row in prose):
            continue
        kept.setdefault(first, []).append(text_of)
    last_row = max((token.end[0] for token in tokens), default=0)
    return {row: kept.get(row, []) for row in range(1, last_row + 1)}


def criterion_references_in_text(text, origin):
    """Resolve every `criterion ...` mention in `text` to an identifier.

    Returns `(references, unresolved)`: `references` maps an identifier to the
    places that name it, and `unresolved` lists the mentions whose identifier
    could not be read at all. The second half is what keeps a new phrasing from
    being skipped: `criterion C6b` and `criterion list item 9` are references
    like any other, and a mention this parser cannot read is a failure rather
    than silence.
    """
    references = {}
    unresolved = []
    for number, pieces in sorted(current_scan_stream()(text).items()):
        line = " ".join(pieces)
        for mention in CRITERION_MENTION.finditer(line):
            tail = CRITERION_REFERENCE_PREFIX.sub("", line[mention.end() :])
            bare = re.match(r"^(\d+)\b", tail)
            named = CRITERION_NAMED.match(tail)
            item = CRITERION_ITEM.match(tail)
            if named:
                identifier = f"C{named.group(1)}{named.group(2)}"
            elif item:
                identifier = f"C{item.group(1)}"
            elif bare:
                # A number without the `C`: a reference this parser cannot
                # place. Reported, because pointing at "criterion 7" is exactly
                # the habit that produced the references this gate guards.
                identifier = None
            elif CRITERION_LIKE.match(tail):
                identifier = None
            else:
                # Prose about the criteria in general, not a reference to one.
                continue
            where = f"{origin}:{number}"
            if identifier is None:
                unresolved.append(f"{where}: {line.strip()!r}")
            else:
                references.setdefault(identifier, []).append(where)
    return references, unresolved


def referenced_criteria(package_dir=PACKAGE_DIR):
    """Criterion identifiers the package's error messages point readers at."""
    references = {}
    unresolved = []
    for dirpath, _, filenames in os.walk(package_dir):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            origin = os.path.relpath(path, package_dir)
            found, unreadable = criterion_references_in_text(text, origin)
            unresolved.extend(unreadable)
            for identifier, where in found.items():
                references.setdefault(identifier, []).extend(where)
    return references, unresolved


class CriterionReferenceTest(unittest.TestCase):
    """A criterion an error message names must be one the guide defines.

    The registry rejects a checkpoint by telling the reader which criterion it
    fails, so a reference to a criterion the guide does not define sends that
    reader looking for something that is not there. The check covers the
    reference forms the code has actually used — `C6b`, `list item 9` — instead
    of only the well-formed `C<n>`, and it reads every place the text can sit,
    including an f-string's literal pieces.
    """

    def test_every_referenced_criterion_is_defined_in_the_guide(self):
        defined = defined_criteria(load_guide())
        self.assertTrue(defined, "the guide defines no criteria at all")
        references, unresolved = referenced_criteria()
        self.assertEqual(
            unresolved,
            [],
            "the code mentions a criterion this gate cannot read; write it as"
            " `criterion C<n>` or `criterion list item <n>` so it can be"
            " checked: " + "; ".join(unresolved),
        )
        missing = {
            number: where
            for number, where in references.items()
            if number not in defined
        }
        self.assertEqual(
            missing,
            {},
            "the code points at criteria the guide does not define: "
            + "; ".join(
                f"{number} ({', '.join(where)})" for number, where in missing.items()
            ),
        )

    def test_an_identifier_like_mention_that_cannot_be_read_is_reported(self):
        # Control, side one: an identifier-looking tail the parser cannot read
        # is a failure, not silence.
        for text in (
            'raise ValueError("chain (MODELS.md, criterion list of them)")\n',
            # The shape a long message splits into: the number is left on the
            # next source line, which is exactly the pre-fix `criterion list` /
            # `item 10` pair.
            'raise ValueError(\n    "chain (MODELS.md, criterion list "\n    "of them"\n)\n',
        ):
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(references, {}, text)
            self.assertEqual(len(unresolved), 1, text)

    def test_prose_about_the_criteria_is_not_a_reference(self):
        # Control, side two: prose is skipped rather than reported, so the gate
        # does not turn every sentence about the criteria into noise.
        for text in (
            "# The guide section that defines the acceptance criteria a",
            "CRITERIA_SECTION = 'the acceptance criteria in \"Adding a new MTP\"'",
            '"criteria and of the fields lives in MODELS.md"',
        ):
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(references, {}, text)
            self.assertEqual(unresolved, [], text)

    def test_one_letter_of_prefix_does_not_hide_a_reference(self):
        # Control for the f-string blind spot: the same message, with and
        # without the `f`. A reader that collects only `STRING` tokens sees the
        # first and not the second where the tokenizer splits the literal text
        # out of it. Both have to be read, on either layout.
        plain = 'raise ValueError("chain (MODELS.md, criterion C9)")\n'
        formatted = 'raise ValueError(f"{family!r} (MODELS.md, criterion C9)")\n'
        mixed = 'raise ValueError(f"{family!r}" " (MODELS.md, criterion C9)")\n'
        for label, text in (
            ("plain string", plain),
            ("f-string", formatted),
            ("f-string plus a plain piece", mixed),
        ):
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(unresolved, [], label)
            self.assertEqual(list(references), ["C9"], label)

    def test_a_broken_reference_inside_an_f_string_is_reported(self):
        # The other side: the f-string's text is not merely collected, it is
        # judged. `C9` is not a criterion the guide defines, so the same edit
        # that is caught in a plain string has to be caught here too.
        formatted = 'raise ValueError(f"{family!r} (MODELS.md, criterion C9)")\n'
        references, unresolved = criterion_references_in_text(formatted, "probe.py")
        defined = defined_criteria(load_guide())
        missing = {name for name in references if name not in defined}
        self.assertEqual(missing, {"C9"})

    def test_the_reader_survives_an_interpreter_without_pep701(self):
        # The guard has to run on the interpreters this project builds on, and
        # `FSTRING_MIDDLE` only exists from Python 3.12. The control models the
        # older layout (an f-string as one `STRING`) and the older capability
        # (the probe is absent, which is what importing the module there leaves
        # it as), then runs the delivered reader unchanged: both reference kinds
        # still have to be read, because on that layout the `STRING` branch is
        # the one that carries them.
        plain = 'raise ValueError("chain (MODELS.md, criterion C9)")\n'
        formatted = 'raise ValueError(f"{family!r} (MODELS.md, criterion C9)")\n'
        # `__name__` is this module under unittest, and something else when a
        # harness exec's the file; either way the globals of the running copy
        # are the ones to patch.
        live = sys.modules.get(__name__)
        namespace = live.__dict__ if live is not None else globals()
        real_tokens, real_probe = namespace["_tokens"], namespace["FSTRING_MIDDLE"]
        # The tokenizer and the probe are switched together: the layout and the
        # capability belong to the same interpreter, so a test that swaps one has
        # to swap the other or it models an interpreter that cannot exist.
        namespace["_tokens"] = tokens_without_pep701
        namespace["FSTRING_MIDDLE"] = (
            None if real_probe is None else tokenize.FSTRING_MIDDLE
        )
        try:
            if real_probe is not None:
                # Where the interpreter has the split layout, show that the
                # model really differs from it: on 3.12 the f-string is split,
                # in the model it is a single STRING. On an interpreter without
                # the split there is nothing to contrast, and the loop below is
                # the whole control.
                kinds = [token.type for token in real_tokens(formatted)]
                modelled = [token.type for token in tokens_without_pep701(formatted)]
                self.assertIn(real_probe, kinds)
                self.assertEqual(modelled.count(tokenize.STRING), 1)
                self.assertNotIn(real_probe, modelled)

            for label, text in (("plain string", plain), ("f-string", formatted)):
                references, unresolved = criterion_references_in_text(text, "probe.py")
                self.assertEqual(unresolved, [], label)
                self.assertEqual(list(references), ["C9"], label)
        finally:
            namespace["_tokens"] = real_tokens
            namespace["FSTRING_MIDDLE"] = real_probe

    def test_the_pre_fix_reference_forms_are_read(self):
        # The three forms the first version of this gate was blind to. They are
        # parsed here from the shapes the code used, so a regression to them is
        # caught by the test above rather than passing silently.
        forms = {
            "criterion C6b": "C6b",
            "criterion list item 9": "C9",
            "criterion list item 10": "C10",
            "criterion C4": "C4",
            "criterion C1: the checkpoint must publish": "C1",
        }
        for text, expected in forms.items():
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(unresolved, [], text)
            self.assertEqual(list(references), [expected], text)

    def test_a_number_without_the_c_prefix_is_reported_not_skipped(self):
        # The forms the second version still walked past: a bare number, and a
        # number behind a separator. Neither can be resolved to a criterion, and
        # both are attempts to point at one, so they fail the gate rather than
        # disappearing.
        # The probes are messages, because that is where a reference lives.
        for fragment in ("criterion 7", "criterion #7"):
            text = f'raise ValueError("chain (MODELS.md, {fragment})")\n'
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(references, {}, text)
            self.assertEqual(len(unresolved), 1, text)
        # Separators a writer reaches for are read, not rejected.
        for fragment, expected in (
            ("criterion: C7", "C7"),
            ("criterion (C7)", "C7"),
            ("criterion - C7", "C7"),
        ):
            text = f'raise ValueError("chain (MODELS.md, {fragment})")\n'
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(unresolved, [], text)
            self.assertEqual(list(references), [expected], text)

    def test_docstrings_and_comments_are_not_scanned(self):
        # Control: prose must not turn into a failure. The multi-line docstring
        # is the case the first version scanned as code.
        prose = {
            "function docstring": 'def f():\n    """About criteria list of them\n    and criteria list item 3.\n    """\n    return 1\n',
            "module docstring": '"""About criteria list of them\nover two lines.\n"""\nx = 1\n',
            "single-line docstring": 'def f():\n    """About criteria list of them."""\n    return 1\n',
            "comment": "# see criteria list item 3 above\nx = 1\n",
        }
        for label, text in prose.items():
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(references, {}, label)
            self.assertEqual(unresolved, [], label)

    def test_a_message_string_is_still_scanned(self):
        # The other side of the same rule: a string that is not a docstring is
        # code, so a reference inside one is seen however the text is laid out.
        for label, text in (
            ("one line", 'raise ValueError("about criterion C4")\n'),
            ("several strings", 'raise ValueError("about criterion C4", "and more")\n'),
        ):
            references, unresolved = criterion_references_in_text(text, "probe.py")
            self.assertEqual(unresolved, [], label)
            self.assertEqual(list(references), ["C4"], label)

    def test_a_reference_split_mid_identifier_is_indistinguishable_from_prose(self):
        # The bounded limitation, recorded as a control so it is a known edge
        # rather than a surprise. When the identifier is its own string piece,
        # the text reads exactly like a sentence ending in the word
        # "criterion", so the reader treats it as prose and skips it.
        text = 'raise ValueError(\n    "about criterion "\n    "C4"\n)\n'
        references, unresolved = criterion_references_in_text(text, "probe.py")
        self.assertEqual(references, {})
        self.assertEqual(unresolved, [])
        # Its realistic shape — the piece that carries the word also carries
        # what follows it — is reported instead (see the pre-fix code, whose
        # `criterion list "` / `item 10` pair looked exactly like this).


# The two markings the field reference uses for fields this build does not
# execute.
DESCRIPTIVE_ONLY_FIELDS = ("embedding_at_position_zero", "recycle_hidden")
# Fields that are read only to be rejected, so a description that sets one fails
# at construction.
RECORDED_FIELDS = (
    "runtime_depth",
    "runtime_depth_keys",
    "shared_layer_depths",
)
# Values that trip each recorded field's rejection.
RECORDED_FIELD_TRIP = {
    "runtime_depth": 1,
    "runtime_depth_keys": ("num_spec",),
    "shared_layer_depths": (0,),
}


def package_files(package_dir=PACKAGE_DIR):
    """The package's Python files, without byte-code caches."""
    for dirpath, _, filenames in os.walk(package_dir):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def package_source(package_dir=PACKAGE_DIR):
    """Concatenated source of the package, without byte-code caches."""
    return "\n".join(
        open(path, encoding="utf-8").read() for path in package_files(package_dir)
    )


def field_reads(names, package_dir=PACKAGE_DIR):
    """Places the package reads any name in `names`, found on the syntax tree.

    Comments and string text are not reads, so mentioning a field in a comment
    does not count; `getattr(spec, "<name>")` and `vars(spec)["<name>"]` are
    reads, so a spelling the previous string search missed does not escape. A
    name is reported when it appears as an attribute (`something.<name>`) or as
    a constant string, which is what the reflective forms pass.
    """
    wanted = set(names)
    found = {}
    for path in package_files(package_dir):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            tree = ast.parse(text)
        except SyntaxError as error:  # pragma: no cover - a broken package
            raise AssertionError(f"{path} does not parse: {error}") from error
        origin = os.path.relpath(path, package_dir)
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute) and node.attr in wanted:
                name = node.attr
            elif isinstance(node, ast.Constant) and node.value in wanted:
                name = node.value
            if name is not None:
                # A field name is also the name of the dataclass attribute it is
                # declared as; the declaration itself is not a read.
                found.setdefault(name, []).append(f"{origin}:{node.lineno}")
    return {name: [where for where in places] for name, places in found.items()}


class DescriptionFieldExecutionTest(unittest.TestCase):
    """The field reference's markings match what the package does with them."""

    def test_a_descriptive_only_field_is_read_by_nothing(self):
        # Read from the syntax tree, so a reader that reaches the field through
        # `getattr` is still a reader, and a comment that names it is not.
        reads = field_reads(DESCRIPTIVE_ONLY_FIELDS)
        for field in DESCRIPTIVE_ONLY_FIELDS:
            self.assertEqual(
                reads.get(field, []),
                [],
                f"{field} is marked descriptive only in MODELS.md but the"
                " package reads it; either execute it and re-mark the guide,"
                " or drop the read",
            )
            self.assertIn(f"`{field}`", load_guide())

    def test_a_recorded_field_fails_at_construction(self):
        from dataclasses import replace as dataclass_replace

        from infinilm.draft_spec import _check_description

        base = next(
            spec for spec in list_draft_model_specs() if spec.family == "qwen3_5_mtp"
        )
        for field, trip in RECORDED_FIELD_TRIP.items():
            self.assertIn(field, RECORDED_FIELDS)
            broken = dataclass_replace(base, **{field: trip})
            with self.assertRaises(
                Exception, msg=f"{field} is marked (recorded) but did not fail"
            ):
                _check_description(broken)

    def test_every_field_is_either_executed_or_marked(self):
        # A field that is neither read nor marked would be a silent no-op, which
        # is exactly what the guide promises not to have. Declaring is not
        # reading: a field that appears only as the dataclass attribute has no
        # place here, and that is what the check catches.
        unmarked = [
            field.name
            for dataclass in (DraftModelSpec, DraftWeightMap)
            for field in fields(dataclass)
            if field.name not in DESCRIPTIVE_ONLY_FIELDS
            and field.name not in RECORDED_FIELDS
        ]
        reads = field_reads(unmarked)
        for name in unmarked:
            self.assertTrue(
                reads.get(name),
                f"{name} is read by nothing and is not marked in MODELS.md",
            )

    def test_the_field_read_check_sees_reflective_reads_and_ignores_comments(self):
        # Controls for the two ways the previous string search went wrong. Both
        # are parsed from source text, so no file has to be touched.
        positive = 'value = getattr(spec, "recycle_hidden")\n'
        negative = "# spec.recycle_hidden is not read here\nvalue = 1\n"
        directory = tempfile.mkdtemp(prefix="docgate_reads_")
        try:
            with open(os.path.join(directory, "probe_positive.py"), "w") as f:
                f.write(positive)
            reads = field_reads(("recycle_hidden",), directory)
            self.assertTrue(
                reads.get("recycle_hidden"),
                "a getattr read escaped the check",
            )
            with open(os.path.join(directory, "probe_positive.py"), "w") as f:
                f.write(negative)
            reads = field_reads(("recycle_hidden",), directory)
            self.assertEqual(
                reads.get("recycle_hidden", []),
                [],
                "a comment naming the field was reported as a read",
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)


class TranslatedGuideTest(unittest.TestCase):
    """The translated guide says which version is authoritative.

    The two files were near 1:1 translations before the description and
    speculative-decoding sections were added to the English one, so a reader of
    the Chinese file needs to be told which document to trust rather than
    discovering the gap by comparing them.
    """

    def declaration(self):
        """The sentence about authority and the text it sits inside."""
        with open(MODELS_ZH_PATH, encoding="utf-8") as f:
            translated = f.read()
        for sentence in translated.split("。"):
            if "权威" in sentence:
                return sentence
        return None

    def test_the_translated_guide_defers_to_the_english_one(self):
        sentence = self.declaration()
        self.assertIsNotNone(sentence, "the translated guide declares no authority")
        # The declaration has to name the English file as the authoritative one,
        # and it has to say so rather than say the opposite. Substring presence
        # is not enough (a sentence naming the other file as authoritative has
        # the same words), and neither is word order alone: a sentence can put
        # `MODELS.md` first and still deny it. So the order is checked, the text
        # between the two anchors has to be the empty connector, and the
        # sentence may not carry a negation.
        self.assertIn(AUTHORITY_PHRASE, sentence)
        english = sentence.index("MODELS.md")
        authoritative = sentence.index(AUTHORITY_PHRASE)
        self.assertLess(
            english,
            authoritative,
            f"the declaration does not make MODELS.md the authoritative one:"
            f" {sentence!r}",
        )
        between = sentence[english + len("MODELS.md") : authoritative]
        # An annotation that only names the file again is allowed; anything else
        # between the two anchors is what a denial needs to sit in.
        residue = between
        for annotation in AUTHORITY_ANNOTATIONS:
            residue = residue.replace(annotation, "")
        self.assertEqual(
            residue,
            "",
            f"the declaration puts {between!r} between MODELS.md and"
            f" {AUTHORITY_PHRASE!r}, which is not the plain claim: {sentence!r}",
        )
        negations = [word for word in NEGATION_WORDS if word in sentence]
        self.assertEqual(
            negations,
            [],
            f"the sentence about authority is negated by {negations}: {sentence!r}",
        )
        # And the declaration has to be about this file deferring, so the
        # sentence body still has to read as the English version being the one
        # to trust.
        self.assertNotIn("本文件为权威", sentence)

    def declarations_that_must_be_rejected(self):
        """Sentences that claim, deny or dodge the authority in different ways."""
        return {
            "reversed": "> **本文件为权威版本，`MODELS.md` 仅为参考译本。**",
            "negated but ordered": (
                "> **本文件并非以 `MODELS.md` 为权威，而是以中文版为准。**"
            ),
            "dodges with punctuation": "> **`MODELS.md`？为权威版本。**",
            "ordered but about the wrong file": (
                "> **`MODELS.md` 是参考译本，中文版为权威版本。**"
            ),
        }

    def test_a_reversed_declaration_would_fail_this_check(self):
        # Control: every one of these carries both substrings, and the first two
        # also put `MODELS.md` before the phrase, so only the stricter reading
        # rejects them.
        for label, sentence in self.declarations_that_must_be_rejected().items():
            with self.subTest(label):
                with self.assertRaises(
                    AssertionError,
                    msg=f"a {label} declaration must not satisfy this check",
                ):
                    self.assert_authority(sentence)

    def assert_authority(self, sentence):
        """The body of the check above, callable on a candidate sentence."""
        self.assertIn(AUTHORITY_PHRASE, sentence)
        english = sentence.index("MODELS.md")
        authoritative = sentence.index(AUTHORITY_PHRASE)
        self.assertLess(english, authoritative)
        between = sentence[english + len("MODELS.md") : authoritative]
        residue = between
        for annotation in AUTHORITY_ANNOTATIONS:
            residue = residue.replace(annotation, "")
        self.assertEqual(residue, "")
        negations = [word for word in NEGATION_WORDS if word in sentence]
        self.assertEqual(negations, [])
        self.assertNotIn("本文件为权威", sentence)

    def test_the_live_declaration_passes_the_same_check(self):
        # The check is not vacuous: the declaration in the file goes through it.
        self.assert_authority(self.declaration())


# The message `explain_missing_draft` produces for a model type no description
# covers. The guide quotes its opening, so both halves are pinned: the literal
# text and the fact that the message *starts* with it.
REJECTION_MESSAGE_OPENS = "model type"
REJECTION_MESSAGE_BODY = "has no draft description"
# The guide's wording around the quote: this is the word that makes it a claim
# about where the text sits, not only about its presence.
GUIDE_QUOTE_OPENS = "opens"


class QuotedMessageTest(unittest.TestCase):
    """A message the guide quotes back to the reader has to exist in the code.

    The guide quotes the opening of the message a rejected checkpoint produces.
    A quote is a claim about the code, and a claim about where the text sits is
    only checkable against the message itself, so the check builds the real one
    instead of looking for a substring.
    """

    def real_message(self):
        import tempfile

        from infinilm.draft_spec import explain_missing_draft

        return explain_missing_draft(tempfile.mkdtemp(prefix="docgate_"), "uncovered")

    def test_the_quoted_message_starts_with_the_opening_the_guide_claims(self):
        message = self.real_message()
        self.assertTrue(
            message.startswith(REJECTION_MESSAGE_OPENS),
            f"the real message no longer opens with {REJECTION_MESSAGE_OPENS!r}:"
            f" {message[:60]!r}",
        )
        self.assertIn(REJECTION_MESSAGE_BODY, message)

        guide = load_guide()
        for fragment in (REJECTION_MESSAGE_OPENS, REJECTION_MESSAGE_BODY):
            self.assertIn(fragment, guide)
        # The guide's own word for the position has to match what the message
        # does; a "begins <middle fragment>" claim fails here.
        for line in guide.splitlines():
            if GUIDE_QUOTE_OPENS in line and REJECTION_MESSAGE_BODY in line:
                opening = line.split("`")[1]
                self.assertTrue(
                    opening.startswith(REJECTION_MESSAGE_OPENS),
                    "the guide says the message opens with"
                    f" {opening!r}, but the real message opens with"
                    f" {message[: len(opening)]!r}",
                )
                self.assertIn(REJECTION_MESSAGE_BODY, opening)
                break
        else:
            self.fail(
                f"the guide no longer quotes the rejection message with"
                f" {GUIDE_QUOTE_OPENS!r}"
            )

    def test_the_wrong_opening_would_fail_this_check(self):
        # Control: the claim the first version of this gate accepted. A message
        # does not open with its middle fragment, so the same assertion applied
        # to `has no draft description` must fail.
        message = self.real_message()
        self.assertFalse(message.startswith(REJECTION_MESSAGE_BODY))
        self.assertTrue(message.startswith(REJECTION_MESSAGE_OPENS))


if __name__ == "__main__":
    unittest.main()
