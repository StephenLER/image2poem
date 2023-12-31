## Image2Poem

### 项目说明

机器学习0530期末大项目代码仓库

项目成员：

* 2021111461 李嘉洋
* 2021110231 梁剑

### 文件路径

```sh
│  BERTMatching.py #微调的bert，返回诗词的匹配
│  embedding.npy #词库中的所有诗词的embedding
│  finetune.ipynb #微调bert代码
│  image_process.py #调用blip2模型
│  view.py #gradio页面
│  langchainModel.py #利用langchain调用大模型api
│  fview.py #gradio页面
|  项目展示.mp4
│
├─BERT_CCPoem_v1
│  │  gen_vec_rep.py
│  │
│  ├─BERT_CCPoem_v1 #开源大模型
│  │      tokenizer_config.json
│  │      special_tokens_map.json
│  │      config.json
│  │      training_args.bin
│  │      vocab.txt
│  │      pytorch_model.bin
│  │      bert_model_params.pth #微调后bert模型的参数
│  │
│  ├─CCPM-master #微调使用数据集
│  │      CCPM_report.pdf
│  │      Readme.md 
│  │      test_public.jsonl
│  │      train.jsonl
│  │      valid.jsonl
│  │
│  └─dataprocess
│          translate.ipynb #数据处理脚本
│          unique_choices.csv
│          translated_choices_data.csv
│          train.csv
│          data.csv
│          choices.csv #选项数据
│
├─Helsinki-NLP---opus-mt-en-zh #翻译模型
│
├─Datasets-master
│  └─CCPC #匹配所用的数据局集
│          README.md
│          ccpc_test_v1.0.json
│          ccpc_train_v1.0.json
│          ccpc_valid_v1.0.json
│          process.ipynb #数据处理脚本
│          choices.csv
│
├─photo #测试图片
├─Lora模型 #提供的Lora古风生成模型
│      oldscene_v1.0.safetensors
```

注意由于本项目使用的是`BLIP2`模型，由于模型体量过大，在使用时需要在本地自行下载模型

```sh
git lfs install
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
```

并修改`image_process.py`的BLIP目录



### 项目启动方式

在终端输入 `python fview.py` 即可在生成的`gradio`页面中启动本项目