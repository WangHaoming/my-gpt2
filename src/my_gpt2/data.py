import torch
from torch.utils.data import Dataset


class TinyTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    字符级语言模型的训练数据集。

    核心思路：给定一段连续 token 序列，以滑动窗口方式切出样本对：
      - 输入 x：第 i 到第 i+block_size-1 个 token
      - 目标 y：第 i+1 到第 i+block_size 个 token（x 整体右移一位）

    这样模型学习的任务是：给定前 N 个 token，预测每个位置的下一个 token。

    例如 block_size=4，文本 token 为 [1,2,3,4,5,6]：
      - 第 0 个样本：x=[1,2,3,4]，y=[2,3,4,5]
      - 第 1 个样本：x=[2,3,4,5]，y=[3,4,5,6]
    """

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        # token_ids 必须比 block_size 长，否则连一个完整样本都切不出来
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must be longer than block_size")

        # 把整个语料转成 long 类型的 tensor，方便后续作为 token ID 送入 Embedding 层
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        # 可以切出的样本总数
        # 每个样本需要 block_size+1 个连续 token（x 用前 block_size 个，y 用后 block_size 个）
        # 最后一个样本从下标 len-block_size-1 开始，所以总数 = len - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 取出从 idx 开始、长度为 block_size+1 的片段
        # 例如 idx=0, block_size=4 → chunk = data[0:5]，共 5 个 token
        chunk = self.data[idx : idx + self.block_size + 1]

        # x：去掉最后一个 token，作为模型输入，形状 (block_size,)
        # y：去掉第一个 token，作为预测目标，形状 (block_size,)
        # x[i] 对应的预测目标是 y[i]，即下一个 token
        return chunk[:-1], chunk[1:]
