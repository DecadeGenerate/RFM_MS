import os
os.system("pip install dask[complete]")
os.system("pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple")
os.system("pip install bcolz")
os.system("pip install regex")
from mindspore import Tensor
from RFM import *
unstructured_knowledge = Tensor([88,   17,  967, 2707, 1906, 9182, 4585,  776,   12, 4845,  184,   43, 1416,  124,  117, 1418,   22, 4048, 3574, 1906,    3])
b_encoder = GenEncoder(1,7,300,256)
data  = {'unstructured_knowledge': unstructured_knowledge}
b_enc_outputs, b_states = b_encoder(data['unstructured_knowledge'])