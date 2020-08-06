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
### 3.2 回译
```
cd LowResource_data_aug/
python augment.py --method bt --input_file data/ori_data/auto_100.csv --output data/aug_data/
```

## 4. 效果验证
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

   | model | acc |
   |:---   |:--- |
   |textCNN    |77.3 |
   |textCNN+BT |81.0 |
   |textCNN+EDA |82.2 |
   |textCNN+EDA+BT | 83.8 |
