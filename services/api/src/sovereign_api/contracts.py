"""Minimal provider-neutral contracts for the first backend slice."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelRequest:
    model_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class ModelResponse:
    model_id: str
    content: str
