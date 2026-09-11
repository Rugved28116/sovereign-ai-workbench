"""Shared generation prompt constraints for API and internal execution."""

from typing import Annotated

from pydantic import StringConstraints

MAX_PROMPT_LENGTH = 32_768
Prompt = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH),
]
