# 项目说明

Simulation code for "End-to-End Supervised Learning for NOMA-enabled Resource Allocation: a Dynamic and Scalable Approach"

```
pip install -r requirements.txt
```


## step 1 Generate data
```shell
python DataOperation.py 
```
## step 2 Train model
using PTN
```shell
python SLPTN.py
```
using GPN
```
python SLGPN.py
```

execute in the background
```
nohup python SLGPN.py > slgpn.log &
nohup python SLPTN.py > slptn.log &
```

results
```
tail -f nohup.out
```