# -*- coding: utf-8 -*-
# MWSC: Multi-scale Wavelet Synchronization Coherence and Deep Learning for Deepfake Detection

import unittest
from check import inspect_source

class SourceTests(unittest.TestCase):
    def test_sources_follow_the_ddin_header_footer_and_spacing_rules(self):
        report = inspect_source()
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["files"], report["python_files"] + report["document_files"])

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
