# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:10:51 2017

@author: abu
"""

import argparse
import os
import time

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

import compressed_sensing as cs
import network as dnn


def undersample(im, gauss_ivar=1e-3):

    mask = cs.cartesian_mask(im.shape, gauss_ivar,
                             centred=False,
                             sample_high_freq=True,
                             sample_centre=True,
                             sample_n=8)

    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')

    return im_und, k_und, mask


def prep_input(im, gauss_ivar=1e-3):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    mask = cs.cartesian_mask(im.shape, gauss_ivar,
                             centred=False,
                             sample_high_freq=True,
                             sample_centre=True,
                             sample_n=8)

    im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')

    im_gnd_l = dnn.complex_to_network(im)
    im_und_l = dnn.complex_to_network(im_und)
    k_und_l = dnn.complex_to_network(k_und)
    mask_l = dnn.complex_to_network(mask, mask=True)

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def load_data():
    """
    Creates dummy dataset from one knee subject for demo.
    In practice, one should take much bigger dataset,
    as well as train & test should have similar distribution.

    Source: http://mridata.org/
    """
    data = loadmat(os.path.join(project_root, './data/lustig_knee_p2.mat'))['xn']
    nx, ny, nz, nc = data.shape

    train = np.transpose(data, (3, 0, 1, 2)).reshape((-1, ny, nz))
    validate = np.transpose(data, (3, 1, 0, 2)).reshape((-1, nx, nz))
    test = np.transpose(data, (3, 2, 0, 1)).reshape((-1, nx, ny))

    return train, validate, test     
     
     
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['16'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate')
    parser.add_argument('--l2', metavar='float', nargs=1,
                        default=['1e-3'], help='l2 regularisation')
    parser.add_argument('--gauss_ivar', metavar='float', nargs=1,
                        default=['0.0005'],
                        help='Sensitivity for Gaussian Distribution which'
                        'decides the undersampling rate of the Cartesian mask')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true',
                        help='Save output images and masks')

    args = parser.parse_args()

    # Project config
    model_name = 'd2_c2'
    gauss_ivar = float(args.gauss_ivar[0])  # undersampling rate
    l2_reg = float(args.l2[0])
    lr_rate = float(args.lr[0])
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny = 128, 128
    save_fig = args.savefig
    save_every = 5

    # Compute acceleration rate
    dummy_mask = cs.cartesian_mask((Nx, Ny), 
                                   gauss_ivar,
                                   sample_high_freq=True,
                                   sample_centre=True, 
                                   sample_n=8)
    acc_rate = dummy_mask.size / np.sum(dummy_mask)
    print('Acceleration Rate: {:.2f}'.format(acc_rate))
    
    # Configure directory info
    project_root = '.'
    save_dir = os.path.join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    train_data, valid_data, test_data = load_data()
    
    shape = dnn.complex_to_network(test_data).shape[1:]
    model = dnn.build_res_model(shape, l2_reg)
    
    # Training
    im_und, k_und, mask, im_gnd = prep_input(train_data, 
                                             gauss_ivar=gauss_ivar)  
    
    t_start = time.time()    
    model.fit([im_und, k_und, 1-mask], im_gnd,
              batch_size=batch_size,
              epochs=num_epoch,
              shuffle=True)
    t_end = time.time()
    print(" time: {:.3f}s".format(t_end - t_start))   
    
    # Test
    im_und_test, k_und_test, mask_test, im_gnd_test = prep_input(test_data, 
                                                                 gauss_ivar=gauss_ivar)      
    score = model.evaluate([im_und_test, k_und_test, 1-mask_test], 
                           im_gnd_test, 
                           verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]) 
    
    # Result
    idx_img = 32
    pre_imgs = model.predict([im_und_test, k_und_test, 1-mask_test])

    plt.set_cmap('bone')   
    
    fig, axes = plt.subplots(2,3, figsize=(9,6))
    img0 = dnn.complex_from_network(im_gnd_test.transpose(0,3,1,2))    
    axes[0,0].imshow(np.abs(img0[idx_img]))

    img1 = dnn.complex_from_network(im_und_test.transpose(0,3,1,2))    
    axes[0,1].imshow(np.abs(img1[idx_img]))
    
    img2 = dnn.complex_from_network(pre_imgs.transpose(0,3,1,2))
    axes[0,2].imshow(np.abs(img2[idx_img]))
    
    img3 = np.abs(img1 - img2)
    axes[1,0].imshow(np.abs(img3[idx_img]))
    
    img4 = np.abs(img0 - img1)
    axes[1,1].imshow(np.abs(img4[idx_img]))    
    
    img5 = np.abs(img0 - img2)
    axes[1,2].imshow(np.abs(img5[idx_img]))    

    plt.savefig('result.png')
    
