from typing import Any
import json


def parse_misparsed(text: str, open: str = None, close: str = None, normalize_newline: bool = True) -> Any:
    """Parse misparsed JSON.
    Args:
        text (str): The text to parse.
        open (str, optional): The open character. Defaults to None.
        close (str, optional): The close character. Defaults to None.
    Returns:
        JSON: The parsed JSON.
    """
    if open is None:
        open = '{"'
    if close is None:
        close = ']}'

    try:
        start = text.index(open)
        end = text.rindex(close) + len(close)
        text = text[start:end]
    except ValueError:
        text = text

    if normalize_newline:
        text = text.replace('\\n', '\n').replace("\n", "\\n")

    return json.loads(text)
