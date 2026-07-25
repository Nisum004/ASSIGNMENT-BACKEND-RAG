from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorDocument:
    id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool]


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, items: list[tuple[str, list[float], dict[str, str | int]]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, vector: list[float], top_k: int) -> list[VectorDocument]:
        raise NotImplementedError

