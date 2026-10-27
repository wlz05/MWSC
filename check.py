# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import ast
import io
import ntpath
import re
from pathlib import Path
import tokenize

HEADER = "# -*- coding: utf-8 -*-"
TITLE = "# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection"
FOOTER = ("# Author: Weiliang Zhu", "# Email: wlzchina05@gmail.com")

def inspect_document(path):
    text = path.read_text(encoding='utf-8')
    errors = []
    if re.search(r'[\u3400-\u9fff]', text):
        errors.append(str(path) + ': documentation must use English')
    if sum(line.startswith('```') for line in text.splitlines()) % 2:
        errors.append(str(path) + ': unclosed Markdown fence')
    for number, line in enumerate(text.splitlines(), 1):
        if line != line.rstrip():
            errors.append(str(path) + ':%d: trailing whitespace' % number)
        if re.search(r'(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/(?:home|Users|root)/)', line):
            errors.append(str(path) + ':%d: use relative local paths' % number)
    return errors

def inspect_source(root="."):
    base = Path(root)
    files = sorted(path for path in base.rglob("*") if path.is_file() and ".git" not in path.relative_to(base).parts)
    errors = []
    documents = []
    line_count = 0
    for path in files:
        if path in (base / "MWSC.md", base / "README.md"):
            documents.append(path.name)
            errors.extend(inspect_document(path))
            continue
        if path.suffix != ".py":
            errors.append(str(path) + ": non-Python file")
            continue
        if re.fullmatch(r"[a-z]+", path.stem) is None:
            errors.append(str(path) + ": filename must be one lowercase English word")
        data = path.read_bytes()
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            errors.append(str(path) + ": non-ASCII content")
            continue
        lines = text.splitlines()
        line_count += len(lines)
        if len(lines) < 6 or lines[:2] != [HEADER, TITLE] or tuple(lines[-2:]) != FOOTER:
            errors.append(str(path) + ": missing DDIN-style header or footer")
        for number, line in enumerate(lines, 1):
            if not line.strip() and line:
                errors.append(str(path) + ":%d: whitespace-only line" % number)
            if not line and (number == 1 or number == len(lines) or not lines[number - 2]):
                errors.append(str(path) + ":%d: misplaced or repeated blank line" % number)
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as error:
            errors.append(str(error))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith(("https://", "http://")):
                    continue
                if value in ("/", "\\", "\\\\"):
                    continue
                if ntpath.isabs(value) and (ntpath.splitdrive(value)[0] or value.startswith(("/", "\\\\"))):
                    errors.append(str(path) + ":%d: absolute path literal" % node.lineno)
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT and len(token.string) > 100:
                errors.append(str(path) + ":%d: comment is not short" % token.start[0])
    return {"files": len(files), "python_files": sum(path.suffix == ".py" for path in files), "document_files": len(documents), "documents": documents, "lines": line_count, "errors": errors, "passed": not errors}

if __name__ == "__main__":
    import json
    report = inspect_source()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
