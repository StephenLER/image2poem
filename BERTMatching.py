import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer,BertConfig
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import logging

bert_path="./BERT_CCPoem_v1/BERT_CCPoem_v1"

gpu_list = None
class CustomBERTModel(nn.Module):
    #定制化BERT函数
    def __init__(self, num_additional_layers=3):
        super(CustomBERTModel, self).__init__()
        # 加载预训练的BERT模型
        self.bert_model = BertModel.from_pretrained(bert_path)
        

    def forward(self, data, cls=False):
        #前馈层
        result = []
        # print(data)
        x = data['input_ids']
        y = self.bert_model(x, attention_mask=data['attention_mask'],
                         token_type_ids=data['token_type_ids'])[0]
        # print("y:",y)
        # print("y:",y.shape)
        
        if(cls):
            result = y[:, 0, :].view(y.size(0), -1)
            result = result.cpu().tolist()
        else:
            result = []
            y = y.cpu()
            # y = torch.mean(y, 1)
            # result = y.cpu().tolist()
            # print("**********")
            for i in range(y.shape[0]):
                # 这里的i代表着批处理
                tmp = y[i][1:torch.sum(data['attention_mask'][i]) - 1, :]
                # print("tmp:",tmp.shape)
                result.append(tmp.mean(0).tolist())

        return result
    
class BertFormatter():
    #BERT数据生成
    def __init__(self, BERT_PATH=bert_path):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    def process(self, data):
        res_dict = self.tokenizer.batch_encode_plus(
            data, pad_to_max_length=True)

        input_list = {'input_ids': torch.LongTensor(res_dict['input_ids']),
                      'attention_mask': torch.LongTensor(res_dict['attention_mask']),
                      "token_type_ids": torch.LongTensor(res_dict['token_type_ids'])}
        return input_list


def init(BERT_PATH=bert_path):
    #初始化
    global gpu_list
    gpu_list = []

    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if(device_list[0] == ""):
        device_list = []
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    cuda = torch.cuda.is_available()
    logging.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logging.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    model = CustomBERTModel(BERT_PATH)
    formatter = BertFormatter(BERT_PATH)
    if len(gpu_list) > 0:
        model = model.cuda()
    if(len(gpu_list) > 1):
        try:
            model.init_multi_gpu(gpu_list)
        except Exception as e:
            logging.warning(
                "No init_multi_gpu implemented in the model, use single gpu instead. {}".format(str(e)))
    return model, formatter


def predict_vec_rep(data, model, formatter):
    #生成embedding的函数
    data = formatter.process(data)
    model.eval()

    for i in data:
        if(isinstance(data[i], torch.Tensor)):
            if len(gpu_list) > 0:
                data[i] = data[i].cuda()
    # print(data)
    result = model(data)

    return result


def cos_sim(vector_a, vector_b, sim=True):
    #计算相似度函数
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    if(not sim):
        return cos
    sim = 0.5 + 0.5 * cos
    return cos

# 创建并初始化模型
custom_bert_model = CustomBERTModel()

# 加载BERT模型的参数字典
bert_state_dict = torch.load('./BERT_CCPoem_v1/BERT_CCPoem_v1/bert_model_params.pth')

# 更新BERT模型的参数
custom_bert_model.bert_model.load_state_dict(bert_state_dict)

#模型初始化
model, formatter = init()

#诗词embedding列表
embedding_array = np.load('embedding.npy')
embedding_array=embedding_array.tolist() 

#诗词列表
df=pd.read_csv('./BERT_CCPoem_v1/dataprocess/choices.csv')
choices=df['choices'].to_list() 


def translation_to_poem(input):
    input_vector = predict_vec_rep([input], model, formatter)[0]
    input_vector = torch.tensor(input_vector).to(device)
    matching_library = torch.tensor(embedding_array).to(device)

    # 执行点乘操作
    similarity_scores = torch.matmul(input_vector, matching_library.t())

    # 获取最相似的前十个答案的索引
    num=10
    top_matches = similarity_scores.topk(k=num).indices
    # top_matches=torch.randperm(num)[:10]

    # 将结果移回CPU（如果需要的话）
    top_matches = top_matches.cpu().numpy()

    # 返回最相似的十个答案
    return [choices[i] for i in top_matches]

