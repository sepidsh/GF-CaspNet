from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
from data import distorted_inputs
import re
from tensorflow.contrib.layers import *
import cv2

import tensorflow.contrib.layers as layers

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from tensorflow import transpose
from tensorflow import multiply
from tensorflow import nn
from tensorflow.python.ops import nn
from tensorflow.contrib import slim
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers, utils
#import tflearn
#from tflearn import get_training_mode







TOWER_NAME = 'tower'

def select_model(name):
    
    return gabor_age


#def drop_path() : 


#def fractal_block (nlabels, images, pkeep, is_training , num_cols, filter ,kernel_size):



       

growthRate=12






def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):
    if requested_step is not None:

        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def gabor_filter(params,images) :
    ksize=5
    t1 = (cv2.getGaborKernel(**params))
    t1 = tf.expand_dims(t1, 2)
    t1 = tf.expand_dims(t1, 3)
    t2 = (cv2.getGaborKernel(**params))
    t2 = tf.expand_dims(t2, 2)
    t2 = tf.expand_dims(t2, 3)
    t3 = (cv2.getGaborKernel(**params))
    t3 = tf.expand_dims(t3, 2)
    t3 = tf.expand_dims(t3, 3)
    filter=tf.concat([t1, t2,t3],2)
    filter=tf.to_float(filter)
    t1=tf.to_float(t1)
    answer = tf.nn.conv2d(images, filter, strides=[1, 1, 1, 1], padding='SAME')

    return answer
    


#def Conv2D(x, out_channel, kernel_shape,
#           padding='SAME', stride=1,
#           W_init=None, b_init=None,
#           nl=tf.identity, split=1, use_bias=True,
#           data_format='NHWC'):
#answer_multi, 96,7, padding='VALID', 4,  scope='conv1'

def add_layer(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        c = tf.contrib.layers.batch_norm(l)
        #c=l
        c = tf.nn.relu(c)
        c = conv('conv1', c, growthRate, 1,'SAME')
        l = tf.concat([c, l], 3)
    return l

def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = tf.contrib.layers.batch_norm(l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = AvgPooling('pool', l, 2)
    return l


#### depth ??????? 
depth=40
N = int((depth - 4)  / 3)
growRate=12









def gabor_age(nlabels, images, pkeep, is_training):

    weight_decay = 0.0005
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.variable_scope("gabor_age", "gabor_age", [images] ) as scope:

        with tf.contrib.slim.arg_scope(
                [convolution2d, fully_connected],
                weights_regularizer=weights_regularizer,
                biases_initializer=tf.constant_initializer(1.),
                weights_initializer=tf.random_normal_initializer(stddev=0.005),
                trainable=True
                ):
            with tf.contrib.slim.arg_scope(
                    [convolution2d],
                    weights_initializer=tf.random_normal_initializer(stddev=0.01)):
                ksize=3 ### gabor wavelets kernel size you  may use 7 9 instead regarding to your data set
                sigma= 0.75
                gamma = 0.25


                '''params = {'ksize':(ksize, ksize), 'sigma':5, 'theta':0, 'lambd':3,'gamma':0.01, 'psi':0}
                answer_1 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize), 'sigma':7, 'theta':np.pi/4, 'lambd':2,'gamma':0.1, 'psi':0}
                answer_2 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize), 'sigma':5, 'theta':np.pi/2, 'lambd':3,'gamma':0.01, 'psi':0}
                answer_3 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize), 'sigma':7, 'theta':3*np.pi/4, 'lambd':2,'gamma':0.1, 'psi':0}
                answer_4 = gabor_filter(params, images) 

                params = {'ksize':(ksize, ksize),  'sigma':5, 'theta':0, 'lambd':3,'gamma':0.01, 'psi':np.pi}
                answer_5 = gabor_filter(params, images)
                params = {'ksize':(ksize, ksize),  'sigma':7, 'theta':np.pi/4, 'lambd':2,'gamma':0.1, 'psi':np.pi}
                answer_6 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize),'sigma':5, 'theta':np.pi/2, 'lambd':3,'gamma':0.01, 'psi':np.pi}
                answer_7 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize),'sigma':7, 'theta':3*np.pi/4, 'lambd':2,'gamma':0.12,  'psi':np.pi}
                answer_8 = gabor_filter(params, images)'''
                

                params = {'ksize':(ksize, ksize), 'sigma':sigma, 'theta':0, 'lambd':2.5,'gamma':gamma, 'psi':0}
                answer_1 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize), 'sigma':sigma, 'theta':np.pi/4, 'lambd':2.5,'gamma':gamma, 'psi':0}
                answer_2 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize), 'sigma':sigma, 'theta':np.pi/2, 'lambd':2.5,'gamma':gamma, 'psi':0}
                answer_3 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize), 'sigma':sigma, 'theta':3*np.pi/4, 'lambd':2.5,'gamma':gamma, 'psi':0}
                answer_4 = gabor_filter(params, images) 

                params = {'ksize':(ksize, ksize),  'sigma':sigma, 'theta':0, 'lambd':2.5,'gamma':gamma, 'psi':np.pi}
                answer_5 = gabor_filter(params, images)
                params = {'ksize':(ksize, ksize),  'sigma':sigma, 'theta':np.pi/4, 'lambd':2.5,'gamma':gamma, 'psi':np.pi}
                answer_6 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize),'sigma':sigma, 'theta':np.pi/2, 'lambd':2.5,'gamma':gamma, 'psi':np.pi}
                answer_7 = gabor_filter(params, images)

                params = {'ksize':(ksize, ksize),'sigma':sigma, 'theta':3*np.pi/4, 'lambd':2.5,'gamma':gamma,  'psi':np.pi}
                answer_8 = gabor_filter(params, images)
                






                #weights_2 = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
                answer_multi=tf.concat([images,answer_1 , answer_2 , answer_3,answer_4,answer_5,answer_6,answer_7,answer_8],3)
                '''answer_multi= tf.add ( answer_1 ,answer_2)
                answer_multi=tf.add(answer_multi, answer_3)
                answer_multi=tf.add(answer_multi, answer_4)
                answer_multi=tf.add(answer_multi, answer_5)
                answer_multi=tf.add(answer_multi, answer_6)
                answer_multi=tf.add(answer_multi, answer_7)
                answer_multi=tf.add(answer_multi, answer_8)
                answer_multi=tf.add(answer_multi/8, images)'''
                
                input_= convolution2d( answer_multi, 11, [ 1, 1 ],weights_initializer=tf.constant_initializer(1.) ,biases_initializer=None )
                #input2=tf.concat([images , input],3, name='input2' )
                conv1 = convolution2d(input_ ,96, [7,7], [4, 4], padding='VALID',  biases_initializer=tf.constant_initializer(0.), scope='conv1')
                pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
                norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')
                conv2 = convolution2d(norm1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2') 
                pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
                norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
                conv3 = convolution2d(norm2, 384, [3, 3], [1, 1], biases_initializer=tf.constant_initializer(0.), padding='SAME', scope='conv3')
                pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
                flat = tf.reshape(pool3, [-1, 192*8*3*3], name='reshape')
                full1 = fully_connected(flat, 512, scope='full1')
                drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
                full2 = fully_connected(drop1, 512, scope='full2')
                drop2 = tf.nn.dropout(full2, pkeep, name='drop2')


    with tf.variable_scope('output') as scope:
        
        weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)
    return output
