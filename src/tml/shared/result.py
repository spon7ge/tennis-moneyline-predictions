from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, Union

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T
    ok: bool = True


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E
    ok: bool = False


Result = Union[Ok[T], Err[E]]
