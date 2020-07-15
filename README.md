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
├── eda                        # EDA算法实现
│   ├── data                   # 原始数据路径，以及其它数据 
│   ├── utils
│       ├── eda_gen.py         # 输入一条句子，返回增广后的句子集
│   └── augment.py             # 主程序，扩充原始数据集
├── models                     # 验证增强数据效果的模型
│   ├── textCNN                # CNN算法实现
├── main.py                    # 验证不同增广算法生成的数据对分类模型的效果
├── README.md
└── requirements.txt           # 第三方库依赖
```

## 3. 增强脚本
### 3.1 EDA
```
cd eda/
python augment.py --input data/weather_data.txt --output data/eda_weather_data.txt --num_aug 9 --alpha 0.1
# 后三个参数可省略
```

## 4. 效果验证脚本
### 4.1 EDA
```
python main.py --aug_data eda/data/eda_weather_data.txt --model textCNN
```
- 固定同一个原始数据集，使用不同方法获得增强后数据集，并通过同一个分类模型训练验证
- 使用不同分类模型时，同一个方法的增益差异
