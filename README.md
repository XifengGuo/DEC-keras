# Deep Embedding Clustering (DEC)

Keras implementation for ICML-2016 paper:

* Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

## Usage
1. Install [Keras>=2.0.9](https://github.com/fchollet/keras), scikit-learn  
```
pip install keras scikit-learn   
```
2. Clone the code to local.   
```
git clone https://github.com/XifengGuo/DEC-keras.git DEC
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


4. Run experiment on MNIST.   
`python DEC.py --dataset mnist`   
or (if there's pretrained autoencoder weights)  
The DEC model will be saved to "results/DEC_model_final.h5".

5. Other usages.   

Use `python DEC.py -h` for help.

## Results

```
python run_exp.py
```
Table 1. Mean performance over 10 trials. See [results.csv](./results/exp1/results.csv) for detailed results for each trial.  

   |        |     |kmeans|AE+kmeans|  DEC  |  paper    
   :--------|:---:|:----:|:-------:|:-----:|----:
   |mnist   | acc | 53   | 88      | 91    | 84 
   |        | nmi | 50   | 81      | 87    | --
   |fmnist  | acc | 47   | 61      | 62    | --
   |        | nmi | 51   | 64      | 65    | --
   |usps    | acc | 67   | 71      | 76    | --
   |        | nmi | 63   | 68      | 79    | --
   |stl     | acc | 70   | 79      | 86    | --
   |        | nmi | 71   | 72      | 82    | --
   |reuters | acc | 52   | 76      | 78    | 72
   |        | nmi | 31   | 52      | 57    | --


## Autoencoder model

![](autoencoders.png)

## Other implementations

Original code (Caffe): https://github.com/piiswrong/dec   
MXNet implementation: https://github.com/dmlc/mxnet/blob/master/example/dec/dec.py   
Keras implementation without pretraining code: https://github.com/fferroni/DEC-Keras
