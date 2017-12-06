from DEC import DEC
import os, csv
import datasets
from keras.optimizers import SGD
from keras.initializers import VarianceScaling
import numpy as np

expdir='./results/exp1'
if not os.path.exists(expdir):
    os.mkdir(expdir)

logfile = open(expdir + '/results.csv', 'a')
logwriter = csv.DictWriter(logfile, fieldnames=['trials', 'acc', 'nmi', 'ari'])
logwriter.writeheader()

trials=10
for db in ['usps', 'reuters10k', 'stl', 'mnist', 'fmnist']:
    logwriter.writerow(dict(trials=db, acc='', nmi='', ari=''))
    save_db_dir = os.path.join(expdir, db)
    if not os.path.exists(save_db_dir):
        os.mkdir(save_db_dir)

        # load dataset
    from datasets import load_data

    x, y = load_data(db)
    n_clusters = len(np.unique(y))

    init = 'glorot_uniform'
    pretrain_optimizer = 'adam'
    # setting parameters
    if db == 'mnist' or db == 'fmnist':
        update_interval = 140
        pretrain_epochs = 300
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
    elif db == 'reuters10k':
        update_interval = 30
        pretrain_epochs = 50
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')  # [-limit, limit], limit=sqrt(1./fan_in)
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
    elif db == 'usps':
        update_interval = 30
        pretrain_epochs = 50
    elif db == 'stl':
        update_interval = 30
        pretrain_epochs = 10

    # prepare model
    dims = [x.shape[-1], 500, 500, 2000, 10]

    '''Training for base and nosp'''
    results = np.zeros(shape=(trials, 3))
    baseline = np.zeros(shape=(trials, 3))
    metrics0=[]
    metrics1=[]
    for i in range(trials):  # base
        save_dir = os.path.join(save_db_dir, 'trial%d' % i)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        dec = DEC(dims=[x.shape[-1], 500, 500, 2000, 10], n_clusters=n_clusters, init=init)
        dec.pretrain(x=x, y=y, optimizer=pretrain_optimizer,
                     epochs=pretrain_epochs,
                     save_dir=save_dir)
        dec.compile(optimizer=SGD(0.01, 0.9), loss='kld')
        dec.fit(x, y=y,
                update_interval=update_interval,
                save_dir=save_dir)

        log = open(os.path.join(save_dir, 'dec_log.csv'), 'r')
        reader = csv.DictReader(log)
        metrics = []
        for row in reader:
            metrics.append([row['acc'], row['nmi'], row['ari']])
        metrics0.append(metrics[0])
        metrics1.append(metrics[-1])
        log.close()

    metrics0, metrics1 = np.asarray(metrics0, dtype=float), np.asarray(metrics1, dtype=float)
    for t, line in enumerate(metrics0):
        logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2]))
    logwriter.writerow(dict(trials=' ', acc=np.mean(metrics0, 0)[0], nmi=np.mean(metrics0, 0)[1], ari=np.mean(metrics0, 0)[2]))
    for t, line in enumerate(metrics1):
        logwriter.writerow(dict(trials=t, acc=line[0], nmi=line[1], ari=line[2]))
    logwriter.writerow(dict(trials=' ', acc=np.mean(metrics1, 0)[0], nmi=np.mean(metrics1, 0)[1], ari=np.mean(metrics1, 0)[2]))

logfile.close()
