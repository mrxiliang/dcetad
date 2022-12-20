

## Requirements
* Python 3 +
* Pytorch 1.0 +
* sklearn 0.19.1 +
* 
## Training
To replicate the results of the paper on all the data sets:
```
python training.py --n_rots=64 --n_epoch=35 --d_out=64 --ndf=32 --dataset=kddrev
python training.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=arrhythmia 
python training.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=thyroid
python training.py --n_rots=256 --n_epoch=1 --d_out=32 --ndf=8 --dataset=satimage-2
python training.py --n_rots=32 --n_epoch=35 --d_out=64 --ndf=32 --dataset=handoutlines
```
