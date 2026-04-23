from __future__ import annotations  # 支持前向引用  for -> CharTokenizer
from dataclasses import dataclass




@dataclass
class CharTokenizer:
    """
    字符级 Tokenizer：把文本中的每个字符映射成一个整数 ID，也能反向解码。

    这是最简单的 tokenizer 实现，粒度是单个字符（不是单词或子词）。
    例如：'a' → 0, 'b' → 1, ...

    属性：
        stoi: str → int 的映射字典，即"字符 → token ID"
        itos: int → str 的映射字典，即"token ID → 字符"（stoi 的反转）
    """

    stoi: dict[str, int]  # string to int：字符 → token ID
    itos: dict[int, str]  # int to string：token ID → 字符

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """
        从原始文本构建 tokenizer：统计所有出现的字符，排序后分配连续 ID。

        sorted(set(text))：去重后排序，保证每次构建结果一致（确定性）
        """
        chars = sorted(set(text))  # 所有不重复字符，按字母顺序排列

        # 字典推导式：给每个字符分配一个唯一整数 ID
        # enumerate(chars) 产生 (0,'a'), (1,'b'), ... 这样的 (index, char) 对
        stoi = {ch: i for i, ch in enumerate(chars)}

        # 反转映射：把 stoi 的 key/value 对调，得到 ID → 字符
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        """词表大小，即训练文本中不同字符的数量，也是模型 vocab_size 参数的值。"""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """把字符串编码成 token ID 列表。例如 'ab' → [0, 1]"""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """把 token ID 列表解码回字符串。例如 [0, 1] → 'ab'"""
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict[str, dict]:
        """把 tokenizer 序列化为普通 dict，方便存入 checkpoint 文件。"""
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, payload: dict[str, dict]) -> "CharTokenizer":
        """
        从 checkpoint 中保存的 dict 恢复 tokenizer。

        JSON 序列化会把所有 key 变成字符串，所以 itos 的 key 需要手动转回 int。
        """
        stoi = {str(k): int(v) for k, v in payload["stoi"].items()}   # key 保持 str
        itos = {int(k): str(v) for k, v in payload["itos"].items()}    # key 转回 int
        return cls(stoi=stoi, itos=itos)
