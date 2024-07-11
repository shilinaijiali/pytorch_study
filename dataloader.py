import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

# 自构建数据集
# dataset = TensorDataset(torch.arange(45))
# # dataset = TensorDataset(torch.arange(1, 40).unsqueeze(1))  # 确保数据维度正确
# dl = DataLoader(dataset,
#                 batch_size=10,
#                 shuffle=True,
#                 num_workers=0,  # 设置为0避免多线程问题
#                 drop_last=True)
# # 数据输出
# for batch in dl:
#     print(batch)

# 最后一个batch的数据量小于10，被舍弃掉了


# 使用多线程

"""
    如果需要使用多线程数据加载，同时避免 RuntimeError: DataLoader worker (pid(s) ...) exited unexpectedly 错误，可以尝试以下几种方法：
        1、确保代码在 if __name__ == "__main__": 块中运行：
            这在 Windows 系统中尤为重要，以确保子进程能够正确启动。      
        2、减少 num_workers 的数量：
            从较小的值（如 1）开始，然后逐步增加，确保系统能够处理多线程数据加载。
        3、检查数据的完整性：
            确保数据没有损坏，并且能够在单线程模式下正确加载。
        4、确保数据集的每一项能够正确被 DataLoader 处理：
            如果数据集的某些项存在问题，可能会导致子进程崩溃。
        5、捕获和处理子进程中的异常：
            这样可以更好地调试和了解问题的根本原因。
"""


class MyDataset(TensorDataset):
    def __getitem__(self, index):
        try:
            return super(MyDataset, self).__getitem__(index)
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            raise


def main():
    # 自构建数据集
    dataset = MyDataset(torch.arange(1, 40))
    dl = DataLoader(dataset,
                    batch_size=10,
                    shuffle=True,
                    num_workers=2,  # 设置为2，可以根据系统资源调整
                    drop_last=True)

    # 数据输出
    for batch in dl:
        print(batch)


if __name__ == "__main__":
    main()
