# LowResource_data_aug

## 1.环境配置
- 系统环境：`Windows`/`Linux`/`macOS` + `python 3.7.7`
- 依赖包安装
    ```bash
    pip install -r requirements.txt
    ```

## 2. 代码结构
```
.
├── eda                          # EDA算法实现  
│   ├── eda_gen.py               # 输入一条句子，返回增广后的句子集
├── bt                           # Back Translate算法实现  
│   ├── bt_gen.py                # 输入一条句子，返回增广后的句子集
├── mixup                        # Text Mixup算法实现  
│   ├── text_mixup.py            # 复现词和句层面的mixup以及loss
├── cvae                         # CVAE算法实现
│   ├── cvae_gen.py              # 读取中间数据，直接保存增强结果
│   ├── gen_tfrecord.py          # 将中间数据转化为TF Record形式
│   ├── data_helper.py           # 将TFR数据载入图中，提供data handler
│   ├── model_bert.py            # CVAE模型实现，编码解码
│   ├── my_modeling_bert.py      # TF版BERT模型实现
│   ├── beam_search_decoder.py   # beam search解码器
│   ├── utils.py                 # 工具类
│   ├── config.json              # 超参数和路径等
├── data                         # 各类数据路径
│   ├── ori_data                 # 原始数据存储路径
│   ├── aug_data                 # 增强数据存储路径
│   ├── stopwords                # 停用词表存储路径
├── classifiers                  # 不同分类模型的实现,用于实验
│   ├── textCNN.py               # TextCNN模型代码
├── augment.py                   # 主程序，扩充原始数据集
├── dataset.py                   # 处理数据为分类器输入形式
├── eval_aug.py                  # 验证增强效果，包括训练和测试  
├── README.md
└── requirements.txt             # 第三方库依赖
```

## 3. 增强脚本
### 3.1 EDA
```
cd LowResource_data_aug/
python augment.py --method eda --input_file data/ori_data/auto_100.csv --output data/aug_data/ --num_aug 9 --alpha 0.2
# 后两个参数可省略
```
### 3.2 回译（慢，增强一条原始语句20s左右，翻译接口限制）
```
cd LowResource_data_aug/
python augment.py --method bt --input_file data/ori_data/auto_100.csv --output data/aug_data/
```
### 3.3 Text Mixup
- 本算法对词或句的embedding进行mixup，不会产生增强数据，所以直接参见4.3实验效果
- Mixup更类似正则化方法，例如dropout和L2等，使模型适应噪声或新的表示
### 3.4 CVAE
```
cd LowResource_data_aug/
python augment.py --method cvae --input_file data/ori_data/auto_100.csv --output data/aug_data/
```
- 端到端实现cvae增强过程，喂入原始数据集，在指定目录中自动生成增强数据

## 4. 效果验证
Notice: 文本分类任务进行验证，目前实现有`textCNN`

词向量下载地址：[百度网盘下载](https://pan.baidu.com/s/1AmXYWVgkxrG4GokevPtNgA?errmsg=Auth+Login+Sucess&errno=0&ssnerror=0& )，放置于data/
### 4.1 EDA
```
python eval_aug.py --train_file data/ori_data/auto_100.csv --test_file data/ori_data/test.csv
python eval_aug.py --train_file data/aug_data/eda_auto_100.csv --test_file data/ori_data/test.csv
```
- EDA采取10倍增强，原始训练集大小分别为100，500，2000，5000(下列指标均在同一个测试集上计算, accuracy)

   | ori_size | 100 | 500 | 2000 | 5000 |
   |:---       |:--- |:--- |:--- |:---|
   |textCNN    |77.3 |86.8 |90.6 |93.4|
   |textCNN+EDA|82.2 |89.8 |91.2 |93.6|
   |increase   | 4.9 | 3.0 | 0.6 | 0.2|

   (1)EDA对分类效果均有提升；训练数据集越小，EDA增强数据效果越显著

   (2)增加真实数据的效果仍然比增强好，例如原始数据集大小为100时，增加900条增强数据的指标为82.2，但只增加400条真实数据的指标可以达到86.8

- 固定同一个原始数据集，使用不同方法获得增强后数据集，并通过同一个分类模型训练验证
- 使用不同分类模型时，同一个方法的增益差异
### 4.2 回译
```
python eval_aug.py --train_file data/ori_data/auto_100.csv --test_file data/ori_data/test.csv
python eval_aug.py --train_file data/aug_data/bt_auto_100.csv --test_file data/ori_data/test.csv
```
- 由于百度翻译api接口的并发限制，1秒钟最多请求一次，所以只对原始训练集100条的情况进行对比实验，放大倍数仍为10，结果如下：

   | ori_size | 100 |
   |:---   |:--- |
   |textCNN    |77.3 |
   |textCNN+BT |81.0 |
   |textCNN+EDA |82.2 |
   |textCNN+EDA+BT | 83.8 |
### 4.3 Text Mixup
```
python eval_aug.py --train_file data/ori_data/auto_100.csv --test_file data/ori_data/test.csv --mix_up None --rate_mixup 0.5 --alpha 1.0
python eval_aug.py --train_file data/ori_data/auto_100.csv --test_file data/ori_data/test.csv --mix_up word --rate_mixup 0.5 --alpha 1.0
python eval_aug.py --train_file data/ori_data/auto_100.csv --test_file data/ori_data/test.csv --mix_up sen --rate_mixup 0.5 --alpha 1.0
```
- 本方法有三种模式：关闭mixup(None)，word mixup(word)，sentence mixup(sen)
- 按照不同的原始数据集size，观察三种模式的效果：

   | ori_size | 100 | 500 | 2000 | 5000 |
   |:---       |:--- |:--- |:--- |:---|
   |textCNN    |77.3 |86.8 |90.6 |93.4|
   |textCNN+wordMixup|79.6 |87.8 |91.6 |93.2|
   |textCNN+senMixup |77.4 |88.0 |92.0 |93.8|
- 两种mixup方法基本均有提升，word mixup在数据相对较少时的表现更好，sentence mixup在数据相对较多时表现更好
- 总体来讲，比eda和bt略差，但这三种方法都是简单有效的数据增强方法
### 4.4 CVAE
```
python eval_aug.py --train_file data/ori_data/auto_100.csv --test_file data/ori_data/test.csv
python eval_aug.py --train_file data/aug_data/cvae_auto_100.csv --test_file data/ori_data/test.csv
```
- 共采取三种解码方式，分别为greedy，beam search，beam search with top-k
- 100条原始数据，增强后约300条，分类实验结果如下：

   | ori_size | 100 |
   |:---   |:--- |
   |textCNN    |77.3 |
   |textCNN+CVAE |76.8 |
