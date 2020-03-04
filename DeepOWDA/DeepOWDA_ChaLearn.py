import pickle
import gzip
import argparse
import os
import sys
import os.path as osp
import numpy as np
#from keras.datasets import mnist
#from svm_classification import svm_classify
from models import create_model
from keras.optimizers import Adam
from objectives_owda import owda_loss

import scipy.io as scio



#if __name__ == '__main__':
def main(args):
    ############
    # Parameters Section

    # the path to save the final learned features
    #save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = args.out_dim  #10

    # the parameters for training the network
    epoch_num = args.epochs #100
    #batch_size = args.batch_size   #800

    # the regularization parameter of the network
    reg_par = args.reg_par  #1e-5

    # The margin and n_components (number of components) parameter used in the loss function
    # n_components should be at most class_size-1
    margin = args.margin #1.0
    n_components = args.n_components #9
    root = args.root

    # Parameter C of SVM
    #C = 1e-1
    # end of parameters section
    ############

    # Load data
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    traindata_mat = 'traindatafull.mat'
    testdata_mat = 'testdatafull.mat'
    datapath = os.path.join(root,traindata_mat)
    #datapath = ''
    Dismatdata = scio.loadmat(datapath)
    x_train = Dismatdata['traindatafull']
    x_train = x_train.copy(order='C')
    x_train = x_train.astype(np.float32)
    y_train = Dismatdata['trainlabelfull']
    y_train = y_train.copy(order='C')
    y_train = y_train.astype(np.float32)
    Yinter = Dismatdata['Yinter']
    Yinter = Yinter.copy(order='C')
    Yinter = Yinter.astype(np.float32)

    datapath = os.path.join(root,testdata_mat)
    Dismatdata = scio.loadmat(datapath)
    x_test = Dismatdata['testdatafull']
    x_test = x_test.copy(order='C')
    x_test = x_test.astype(np.float32)

    #x_train = np.reshape(x_train, (len(x_train), -1))
    #x_test = np.reshape(x_test, (len(x_test), -1))
    #batch_size = len(x_train)

    L = y_train.shape[1]-1
    yt = y_train[:,L].flatten().astype(np.int64)
    classes = Yinter.shape[0]
    pid = []
    for c in range(classes):
        pids = np.where(yt==c)
        pid.append(pids[0])

    batch_index = np.array([]).astype(np.int64)
    for c in range(classes):
        slices = np.random.choice(pid[c],500)
        batch_index = np.append(batch_index,slices)
    x_train_batch = x_train[batch_index,:]
    y_train_batch = y_train[batch_index,:]
    batch_size = len(x_train_batch)
    print(batch_size)

    # Building, training, and producing the new features by Deep LDA
    model = create_model(x_train.shape[-1], reg_par, outdim_size)

    model_optimizer = Adam()
    model.compile(loss=owda_loss(n_components, margin, Yinter), optimizer=model_optimizer)

    model.summary()

    #model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, shuffle=True, validation_data=(x_test, y_test), verbose=2)
    #model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, shuffle=True, verbose=2)
    model.fit(x_train_batch, y_train_batch, batch_size=batch_size, epochs=1, shuffle=True, verbose=2)

    for ite in range(epoch_num):
        batch_index = np.array([]).astype(np.int64)
        for c in range(classes):
            slices = np.random.choice(pid[c],500)
            batch_index = np.append(batch_index,slices)
        x_train_batch = x_train[batch_index,:]
        y_train_batch = y_train[batch_index,:]
        model.fit(x_train_batch, y_train_batch, batch_size=batch_size, epochs=1, shuffle=True, verbose=2)

    train_feature = model.predict(x_train)
    test_feature = model.predict(x_test)

    #train_feature = train_feature.numpy()
    #test_feature = test_feature.numpy()

    savedata_mat = "TransFeatures.mat";
    savedatapath = os.path.join(root,savedata_mat)
    scio.savemat(savedatapath,{'train_feature':train_feature,'test_feature':test_feature})

    # Training and testing of SVM with linear kernel on the new features
    #[train_acc, test_acc] = svm_classify(x_train_new, y_train, x_test_new, y_test, C=C)
    #print("Accuracy on train data is:", train_acc * 100.0)
    #print("Accuracy on test data is:", test_acc*100.0)

    # Saving new features in a gzip pickled file specified by save_to
    #print('Saving new features ...')
    #f = gzip.open(save_to, 'wb')
    #pickle.dump([(x_train_new, y_train), (x_test_new, y_test)], f)
    #f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Metric Learning')

    # hype-parameters
    parser.add_argument('--reg_par', type=float, default=1e-5, help="learning rate of new parameters")  #1e-5
    #.add_argument('--batch_size', '-b', default=128, type=int, metavar='N',
    #                    help='mini-batch size (1 = pure stochastic) Default: 256')
    
    parser.add_argument('--out_dim', default=512, type=int, metavar='n',
                        help='dimension of embedding space')

    parser.add_argument('--margin', default=1.0, type=float,
                        help='margin in loss function')
    
    parser.add_argument('--root', type=str, default=None,
                        help='path to Data Set')

    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='epochs for training process')
    
    parser.add_argument('--classnum', type=int, default=20)
    parser.add_argument('--n_components', type=int, default=8)


    args = parser.parse_args()
    main(parser.parse_args())
