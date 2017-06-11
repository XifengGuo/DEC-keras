from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model


class SAE(object):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    
    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)
        
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: batch size
    """
    def __init__(self, dims, act='relu', drop_rate=0.2, batch_size=256):
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.stacks = [self.make_stack(i) for i in range(self.n_stacks)]
        self.autoencoders = self.make_autoencoders()
        plot_model(self.autoencoders, show_shapes=True, to_file='autoencoders.png')

    def make_autoencoders(self):
        """ Fully connected autoencoders model, symmetric.
        """
        # input
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], activation=self.activation, name='encoder_%d' % i)(h)

        # hidden layer
        h = Dense(self.dims[-1], name='encoder_%d' % (self.n_stacks - 1))(h)  # features are extracted from here

        # internal layers in decoder
        for i in range(self.n_stacks-1, 0, -1):
            h = Dense(self.dims[i], activation=self.activation, name='decoder_%d' % i)(h)

        # output
        h = Dense(self.dims[0], name='decoder_0')(h)

        return Model(inputs=x, outputs=h)

    def make_stack(self, ith):
        """ 
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is 
        corrupted by Dropout(drop_rate)
        
        Arguments:
            ith: int, in [0, self.n_stacks)
        """
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        output_act = self.activation
        hidden_act = self.activation
        if ith == 0:
            output_act = 'linear'
        if ith == self.n_stacks-1:
            hidden_act = 'linear'
        model = Sequential()
        model.add(Dropout(self.drop_rate, input_shape=(in_out_dim,)))
        model.add(Dense(units=hidden_dim, activation=hidden_act, name='encoder_%d' % ith))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(units=in_out_dim, activation=output_act, name='decoder_%d' % ith))

        plot_model(model, to_file='stack_%d.png' % ith, show_shapes=True)
        return model

    def pretrain_stacks(self, x, epochs=200):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
        """
        features = x
        for i in range(self.n_stacks):
            print 'Pretraining the %dth layer...' % (i+1)
            for j in range(3):  # learning rate multiplies 0.1 every 'epochs/3' epochs
                print 'learning rate =', pow(10, -1-j)
                self.stacks[i].compile(optimizer=SGD(pow(10, -1-j), momentum=0.9), loss='mse')
                self.stacks[i].fit(features, features, batch_size=self.batch_size, epochs=epochs/3)
            print 'The %dth layer has been pretrained.' % (i+1)

            # update features to the inputs of the next layer
            feature_model = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features = feature_model.predict(features)

    def pretrain_autoencoders(self, x, epochs=500):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        :param x: input data, shape=(n_samples, n_dims)
        :param epochs: training epochs
        :return: 
        """
        print 'Copying layer-wise pretrained weights to deep autoencoders'
        for i in range(self.n_stacks):
            name = 'encoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name = 'decoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())

        print 'Fine-tuning autoencoder end-to-end'
        for j in range(epochs/80):
            lr = 0.1*pow(10, -j)
            print 'learning rate =', lr
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=80)

    def fit(self, x, epochs=200):
        self.pretrain_stacks(x, epochs=epochs/2)
        self.pretrain_autoencoders(x, epochs=epochs)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders.
        
        :param x: data
        :return: features
        """
        hidden_layer = self.autoencoders.get_layer(name='encoder_%d' % (self.n_stacks - 1))
        feature_model = Model(self.autoencoders.input, hidden_layer.output)
        return feature_model.predict(x, batch_size=self.batch_size)


if __name__ == "__main__":
    """
    An example for how to use SAE model on MNIST dataset. In terminal run
            python SAE.py
    to see the result. You may get NMI=0.77.
    """
    import numpy as np

    def load_mnist():
        # the data, shuffled and split between train and test sets
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1))
        x = np.divide(x, 50.)  # normalize as it does in DEC paper
        print 'MNIST samples', x.shape
        return x, y

    db = 'mnist'
    n_clusters = 10
    x, y = load_mnist()

    # define and train SAE model
    sae = SAE(dims=[x.shape[-1], 500, 500, 2000, 10])
    sae.fit(x=x, epochs=400)
    sae.autoencoders.save_weights('weights_%s.h5' % db)

    # extract features
    print 'Finished training, extracting features using the trained SAE model'
    features = sae.extract_feature(x)

    print 'performing k-means clustering on the extracted features'
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters, n_init=20)
    y_pred = km.fit_predict(features)

    from sklearn.metrics import normalized_mutual_info_score as nmi
    print 'K-means clustering result on extracted features: NMI =', nmi(y, y_pred)