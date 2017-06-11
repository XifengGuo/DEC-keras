# Deep Embedding Clustering (DEC)

Keras implementation for ICML-2016 paper:

* Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

## Usage
1. Install [Keras v2.0](https://github.com/fchollet/keras), scikit-learn and git   
`sudo pip install keras scikit-learn`   
`sudo apt-get install git`
2. Clone the code to local.   
`git clone https://github.com/XifengGuo/DEC.git DEC`
3. Prepare datasets.    

        cd DEC/data/usps   
        bash ./download_usps.sh   
        cd ../reuters  
        bash ./get_data.sh   
        cd ../..

4. Run experiment on MNIST.   
`python DEC.py mnist`   
or (if there's pretrained autoencoder weights)  
`python DEC.py mnist --ae_weights ae_weights/mnist_ae_weights.h5`   
The DEC model will be saved to "results/DEC_model_final.h5".

5. Run experiment on USPS.   
`python DEC.py usps --update_interval 30`   
or   
`python DEC.py usps --ae_weights ae_weights/usps_ae_weights --update_interval 30`

6. Run experiment on REUTERSIDF10K.   
`python DEC.py reutersidf10k --n_clusters 4 --update_interval 20`   
or   
`python DEC.py reutersidf10k --ae_weights ae_weights/reutersidf10k_ae_weights --n_clusters 4 --update_interval 20`

## Other implementations

Original code (Caffe): https://github.com/piiswrong/dec   
MXNet implementation: https://github.com/dmlc/mxnet/blob/master/example/dec/dec.py   
Keras implementation without pretraining code: https://github.com/fferroni/DEC-Keras