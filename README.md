# 项目说明

```
pip install -r requirements.txt
```


## step 1 生成数据
```shell
python DataOperation.py 
```
用来生成监督学习数据，结果会输出在data文件夹中
2023-04-17-20-14-08是家里电脑生成的1000*7的数据
2023-04-17-21-49-50是家里电脑生成的1000*7的数据后改名为tiny_data
2023-04-18-08-42-37是服务器上生成的10000*7的数据 后改名为small_data
medium_data 50000*7的数据
large_data 100000*7的数据
## step 2 生成模型
使用PTN训练模型
```shell
python SLPTN.py
```
使用GPN训练模型
```
python SLGPN.py
```

后台训练
```
nohup python SLGPN.py > slgpn.log &
nohup python SLPTN.py > slptn.log &
```

查看结果
```
tail -f nohup.out
```