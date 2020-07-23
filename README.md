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
python augment.py --method eda --input_file data/ori_data/weather_data.txt --output data/aug_data/ --num_aug 9 --alpha 0.1
# 后两个参数可省略
```

## 4. 效果验证脚本（调试中）
### 4.1 EDA
```
python eval_aug.py --train_data data/aug_data/eda_weather_data.txt --model textCNN
python eval_aug.py --train_data data/ori_data/weather_data.txt --model textCNN
```
- 固定同一个原始数据集，使用不同方法获得增强后数据集，并通过同一个分类模型训练验证
- 使用不同分类模型时，同一个方法的增益差异
