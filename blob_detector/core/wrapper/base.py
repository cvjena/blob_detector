from __future__ import annotations

from dataclasses import dataclass

@dataclass
class BaseWrapper:

    parent: T.Optional[ImageWrapper] = None
    creator: T.Optional[str] = None
