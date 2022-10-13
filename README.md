# Deep Embedding Clustering (DEC)

Keras implementation for ICML-2016 paper:

* Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

This repo has been modiifed to use custom datasets as well. 
As an example the Vectorization roof dataset(https://github.com/SimonHensel/Vectorization-Roof-Data-Set)
has been used here.

## Usage
1. Install [Keras>=2.0.9](https://github.com/fchollet/keras), scikit-learn  
```
pip install keras scikit-learn   
```
2. Clone the code to local.   
```
git clone https://github.com/guneetmutreja/DEC-keras.git DEC
cd DEC
```
3. Prepare datasets.    

Download **STL**:
```
cd data/stl
bash get_data.sh
cd ../..
```
**MNIST** and **Fashion-MNIST (FMNIST)** can be downloaded automatically when you run the code.

**Reuters** and **USPS**: If you cannot find these datasets yourself, you can download them from:   
https://pan.baidu.com/s/1hsMQ8Tm (password: `4ss4`) for **Reuters**, and  
https://pan.baidu.com/s/1skRg9Dr (password: `sc58`) for **USPS**

Any **Custom** dataset can be provided as a local path in the datasets.py file.


4. Run experiment on CUSTOM.   
`python DEC.py --dataset custom`   
or (if there's pretrained autoencoder weights)  
`python DEC.py --dataset custom --ae_weights ae_weights.h5` 

The DEC model will be saved to "results/DEC_model_final.h5".

5. Other usages.   

Use `python DEC.py -h` for help.



## Autoencoder model

![](autoencoders.png)

## Other implementations

Original code (Caffe): https://github.com/piiswrong/dec   
MXNet implementation: https://github.com/dmlc/mxnet/blob/master/example/dec/dec.py   
Keras implementation without pretraining code: https://github.com/fferroni/DEC-Keras
