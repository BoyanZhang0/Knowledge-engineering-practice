# KE_Exp2
基于Pytorch搭建了一个全连接神经网络(FCNN)模型。
main.py是空的，训练和测试的脚本分别在train和test里。
#### 直接双击.bat文件就能运行。注：需要根据电脑中python环境位置修改bat文件中python环境路径

---
## 文件功能
* **train_A.bat**是训练A组模型并保存；
* **train_B.bat**是训练B组模型并保存；
* **train_A_continue.bat**是加载之前训练的A组模型继续训练并保存；
* **train_B_continue.bat**是加载之前训练的B组模型继续训练并保存；
* **test_A.bat**用于测试A组数据并保存测试结果；
* **test_B.bat**用于测试A组数据并保存测试结果。
---
## 保存位置
* **./Data**是数据文件，train是训练数据，test是测试数据，训练数据会在训练过程中自动划分训练集和验证集；
* **./Fig**是训练结果可视化，文件名后缀是日期时间；
* **./Models**是加载的训练权重，如想加载不同的模型，用记事本打开bat或者用pycharm改对应的bat文件参数就可以了；
* **./testResult**是测试结果，这里的测试集是伪测试集，从训练集中抠出来的。
---
## 运行环境
*python版本：3.9
运行时需要的第三方库：
* 插件：torch 版本：2.1.2+cu118
* 插件：scikit-learn 版本：1.3.0
* 插件：tqdm 版本：4.65.0
* 插件：matplotlib 版本：3.7.2
* 插件：argparse 版本：1.1

运行时需要安装如下环境：
```bash
pip install torch scikit-learn tqdm matplotlib
```
