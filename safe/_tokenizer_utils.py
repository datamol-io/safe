"""Dependency-free SAFE token splitting primitives."""

import re

from loguru import logger


class SAFESplitter:
    """Split a SAFE string into notation tokens."""

    REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%\([0-9]{1,5}\)|\%[0-9]{2}|[0-9])"""

    name = "safe"

    def __init__(self, pattern=None):
        if pattern is None:
            pattern = self.REGEX_PATTERN
        self.regex = re.compile(pattern)

    def tokenize(self, line):
        """Tokenize a SAFE string."""
        if isinstance(line, str):
            tokens = list(self.regex.findall(line))
            reconstruction = "".join(tokens)
            if line != reconstruction:
                logger.error(
                    f"Tokens different from sample:\ntokens {reconstruction}\nsample {line}."
                )
                raise ValueError(line)
        else:
            idxs = re.finditer(self.regex, str(line))
            tokens = [line[match.start(0) : match.end(0)] for match in idxs]
        return tokens

    def detokenize(self, chars):
        """Reconstruct a SAFE string from tokens."""
        if isinstance(chars, str):
            chars = chars.split(" ")
        return "".join(value.strip() for value in chars)

    def split(self, _index, normalized):
        """Pretokenize a value for Hugging Face Tokenizers."""
        return self.tokenize(normalized)

    def pre_tokenize(self, pretok):
        """Apply this splitter to a Hugging Face pretokenizer."""
        pretok.split(self.split)


def split(safe_str: str):
    """Split a SAFE string into notation tokens."""
    return SAFESplitter().tokenize(safe_str)
