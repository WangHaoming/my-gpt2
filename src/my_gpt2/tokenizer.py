from dataclasses import dataclass


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict[str, dict]:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, payload: dict[str, dict]) -> "CharTokenizer":
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}
        itos = {int(k): str(v) for k, v in payload["itos"].items()}
        return cls(stoi=stoi, itos=itos)
