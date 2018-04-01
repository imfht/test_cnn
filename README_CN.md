# test_cnn
test_cnn基于keras，可以用来破解宝塔面板的验证码。基于深度学习的技术，识别效果接近100%准确。

## 开始
跟着步骤一步一步走，很简单就可以直接获得一个准确率超级高的模型。
## 开始之前
因为宝塔控制面板用来生成验证码的代码不支持 Python3，所以这个项目只支持 Python2.7。
## 开始构建 
### 第一步
下载 test_cnn 到本地
```bash
git clone https://github.com/fiht/test_cnn
```

### python 模块依赖
使用`python install -r requirements.txt`安装依赖，推荐您使用virtualenv 构建虚拟环境进行测试。

### 训练模型
使用 ```python run_cnn.py``` 开始训练，默认将训练64个 epoch。在 `run_cnn.py`的第125行可以修改训练的轮数。每轮的训练结果会序列化到 trained.model 文件中，所以如果你训练到第20轮就不想再训练了的话，可以直接 Ctrl-C，然后使用训练过20轮的 trains.model 来进行训练。

### 或者使用我训练好的模型
我已经训练好了我的模型，你可以直接使用`predict.py`预测验证码而不是从头自己训练。

## 想知道更多吗？
访问[我的博客](https://blog.fht.im/15220580960875.html) 获取更多资料。

## 实际效果
![](https://github.com/fiht/test_cnn/raw/master/static/1581522318375_.pic_hd.jpg)

## 网页预览版 [bt.dev.fht.im](https://bt.dev.fht.im) 效果如下
![](https://ws1.sinaimg.cn/large/0062TDWsly1fpxbt7rlp3j327s0w044d.jpg)