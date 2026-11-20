"""Common interface so scripts never care which backend is running."""
from abc import ABC, abstractmethod


class TextGenerator(ABC):
    @abstractmethod
    def generate(self, prompts: list[str], n: int, **sampling) -> list[dict]:
        """Return records: {"prompt", "text", "sample_id", ...}."""

    def close(self) -> None:
        pass
