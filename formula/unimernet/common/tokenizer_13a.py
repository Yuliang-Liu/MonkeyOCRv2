# Source: https://hf-mirror.com/spaces/evaluate-metric/bleu/resolve/main/tokenizer_13a.py
# Original upstream: https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py

import re
from functools import lru_cache


class BaseTokenizer:
    def signature(self):
        return "none"

    def __call__(self, line):
        return line


class TokenizerRegexp(BaseTokenizer):
    def signature(self):
        return "re"

    def __init__(self):
        self._re = [
            (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
            (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
            (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
            (re.compile(r"([0-9])(-)"), r"\1 \2 "),
        ]

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        for regex, repl in self._re:
            line = regex.sub(repl, line)
        return line.split()


class Tokenizer13a(BaseTokenizer):
    def signature(self):
        return "13a"

    def __init__(self):
        self._post_tokenizer = TokenizerRegexp()

    @lru_cache(maxsize=2**16)
    def __call__(self, line):
        line = line.replace("<skipped>", "")
        line = line.replace("-\n", "")
        line = line.replace("\n", " ")

        if "&" in line:
            line = line.replace("&quot;", '"')
            line = line.replace("&amp;", "&")
            line = line.replace("&lt;", "<")
            line = line.replace("&gt;", ">")

        return self._post_tokenizer(f" {line} ")
