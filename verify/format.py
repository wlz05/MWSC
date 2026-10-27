# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import unittest
from pathlib import Path
from check import inspect_source, inspect_document
from verify.fixture import TemporaryWorkspace

class SourceTests(unittest.TestCase):
    def test_sources_follow_the_ddin_header_footer_and_spacing_rules(self):
        report = inspect_source()
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["files"], report["python_files"] + report["document_files"])

    def test_document_checks_detect_nonenglish_notes_and_absolute_paths(self):
        with TemporaryWorkspace() as root:
            path = root / 'guide.md'
            for text in ('# ' + chr(0x4e2d), '`' + 'D:' + chr(92) + 'cache`', '```python', 'text '):
                path.write_text(text, encoding='utf-8')
                self.assertTrue(inspect_document(path))

    def test_document_checks_accept_portable_paths_and_source_links(self):
        with TemporaryWorkspace() as root:
            path = root / 'guide.md'
            path.write_text('# Models\n\n`../runtime/mwsc/syncnet.model`\n\n[Source](https://github.com/wlz05/MWSC)\n', encoding='utf-8')
            self.assertEqual(inspect_document(path), [])

    def test_source_checks_detect_escaped_nonenglish_text(self):
        lines = Path('check.py').read_text(encoding='utf-8').splitlines()
        payload = '\n'.join(lines[:2]) + '\n\nmessage = ' + ascii(chr(0x4e2d)) + '\n\n' + '\n'.join(lines[-2:]) + '\n'
        with TemporaryWorkspace() as root:
            (root / 'sample.py').write_text(payload, encoding='utf-8')
            report = inspect_source(root)
            self.assertTrue(any('source strings must use English' in error for error in report['errors']))

    def test_source_checks_detect_long_docs_and_trailing_spaces(self):
        lines = Path('check.py').read_text(encoding='utf-8').splitlines()
        body = 'def sample():\n    ' + chr(34) * 3 + 'First line.\n    Second line.' + chr(34) * 3 + '\n    return 1 '
        payload = '\n'.join(lines[:2]) + '\n\n' + body + '\n\n' + '\n'.join(lines[-2:]) + '\n'
        with TemporaryWorkspace() as root:
            (root / 'sample.py').write_text(payload, encoding='utf-8')
            report = inspect_source(root)
            self.assertTrue(any('single-line docstring' in error for error in report['errors']))
            self.assertTrue(any('trailing whitespace' in error for error in report['errors']))

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
