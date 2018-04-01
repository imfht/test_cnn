# test_cnn
A keras cnn example: use cnn crack BT Dashboard(宝塔面板) captcha code.

## Getting Started
follow the steps and you will get a nearly 100% acc model of BTDashboard captchacode.
## Prerequisites
Because of BTDashboard only works on python2, so this project only support python2.7, so you should have python2.7 installed.

## Build 
### First
clone this repo.
```bash
git clone https://github.com/fiht/test_cnn
```
### Requirements
All packages we need is at `requirements.txt` , I strongly suggest you use virtualenv manage packages.

### Train model
you can train the model by you self or just use the model I've trained to predict captchacode.

#### Train It yourself
run run_cnn.py to train the model. As default, it will train 64 epoll, you can change it by edit the source code at `cnn.py`  line 125. each model it would save trained result.feel easy to kill it if it can predict well or cost too much time.
  
#### Use trained.model
I've trained the model with 1024 epoll. you can use it directly. Run `python predict.py` to know how to use.

## Wanner know more?
[visit my blog](https://blog.fht.im/15220580960875.html) get more detail.

## Result
![](https://github.com/fiht/test_cnn/raw/master/static/1581522318375_.pic_hd.jpg)


## Other Language
[查看中文版README](https://github.com/fiht/test_cnn/blob/master/README_CN.md)

## a web interface is at [bt.dev.fht.im](https://bt.dev.fht.im)
![](https://ws1.sinaimg.cn/large/0062TDWsly1fpxbt7rlp3j327s0w044d.jpg)