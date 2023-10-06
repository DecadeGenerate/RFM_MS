# 导入所需的模块
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
import numpy as np

# 定义一个生成器函数，用于产生随机的输入和标签数据
def generator_func():
    for i in range(100):
        x = np.random.randn(1, 10)  # 输入数据，形状为(1, 10)
        y = np.random.randint(0, 2, (1,))  # 标签数据，形状为(1,)
        yield (x, y)


# 创建一个GeneratorDataset对象，指定生成器函数和列名
dataset = GeneratorDataset(generator_func, ["x", "y"])

# 对数据集进行一些变换和操作，如打乱、批处理、重复等
dataset = dataset.shuffle(buffer_size=10)  # 随机打乱数据
dataset = dataset.batch(batch_size=4)  # 将数据分批，每批大小为4
dataset = dataset.repeat(count=2)  # 将数据重复两次


# 定义一个简单的全连接网络，用于二分类任务
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Dense(10, 2)  # 输入维度为10，输出维度为2
        self.softmax = nn.Softmax()

    def construct(self, x):
        x = self.flatten(x)  # 将输入展平为一维向量
        x = self.dense(x)  # 通过全连接层得到输出
        x = self.softmax(x)  # 通过softmax函数得到概率分布
        return x


# 创建一个网络实例
model = Network()

# 定义损失函数和优化器
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)  # 使用交叉熵损失函数
optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)  # 使用Adam优化器


# # 定义训练循环
# def train_loop(dataset, model, loss_fn, optimizer):
#     for epoch in range(5):  # 训练5个epoch
#         total_loss = 0  # 记录每个epoch的总损失
#         for data in dataset.create_dict_iterator():  # 通过字典迭代器获取数据
#             x = data["x"]  # 获取输入数据
#             y = data["y"]  # 获取标签数据
#              # 使用梯度上下文管理器计算梯度
#             output = model(x)  # 通过模型得到输出
#             loss = loss_fn(output, y)  # 计算损失值
#             grads = ops.grad(loss, model.trainable_params())  # 计算梯度值
#             optimizer(grads)  # 通过优化器更新参数
#             total_loss += loss.asnumpy()  # 累加损失值
#         print(f"Epoch {epoch}, loss: {total_loss}")  # 打印每个epoch的总损失


# 调用训练循环函数，开始训练模型
# train_loop(dataset, model, loss_fn, optimizer)