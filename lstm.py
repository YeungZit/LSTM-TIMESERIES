#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:45:57 2019

@author: linjunqi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os 
import plotly.offline as py
import plotly.graph_objs as go

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
tf.reset_default_graph()
dir_ = os.getcwd()
data = pd.read_csv(dir_+'/dataset_1.csv')['close'][:6000].tolist()
data.reverse()
data = np.array(data)
data=(data-np.mean(data))/np.std(data)
print("train x and y's shape is batch*steps*1")
print(len(data))


def get_batch():

    global BATCH_START,TIME_STEPS
    trainDataX = data[BATCH_START:BATCH_START+TIME_STEPS*BATCH_SIZE].reshape((BATCH_SIZE,TIME_STEPS))
    trainDataY = data[BATCH_START+1:BATCH_START+TIME_STEPS*BATCH_SIZE+1].reshape((BATCH_SIZE,TIME_STEPS))
    BATCH_START += TIME_STEPS
    return [trainDataX[:,:,np.newaxis],trainDataY[:,:,np.newaxis]]

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_SIZE], name='xs')
    ys = tf.placeholder(tf.float32, [None, TIME_STEPS, OUTPUT_SIZE ], name='ys')

def weight_variable(shape, name='weights'):
    initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
    return tf.get_variable(shape=shape, initializer=initializer,name=name)

def bias_variable(shape, name='biases'):
    initializer = tf.constant_initializer(0.1)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)

def ms_error(labels, logits):
    return tf.square(tf.subtract(labels, logits))

def LSTM(n_steps, input_size, output_size, cell_size, batch_size):
    ##input layer
    with tf.variable_scope('in_hidden'):
        l_in_x = tf.reshape(xs, [-1, input_size], name='2_2D')
        Ws_in = weight_variable([input_size, cell_size])
        bs_in = bias_variable([cell_size,])
        with tf.name_scope('Wx_plus_in_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        l_in_y = tf.reshape(l_in_y, [-1, n_steps, cell_size], name='2_3D')
    
    ##lstm cell
    with tf.variable_scope('LSTM_cell'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            cell_init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                lstm_cell, l_in_y, initial_state=cell_init_state, time_major=False)
    
    ##output layer
    with tf.variable_scope('out_hidden'):
        l_out_x = tf.reshape(cell_outputs, [-1, cell_size], name='2_2D')
        Ws_out = weight_variable([cell_size, output_size])
        bs_out = bias_variable([output_size, ])
        with tf.name_scope('Wx_plus_out_b'):
            pred = tf.matmul(l_out_x, Ws_out) + bs_out
        
#        with tf.name_scope('draw'):
#            drawpred = pred[0]
#            tf.summary.scalar('pred', drawpred)
            
    model={'pred':pred,'final_state':cell_final_state}   
    return model

def train_lstm(pred):
    with tf.name_scope('cost'):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                [tf.reshape(pred, [-1], name='reshape_pred')],
                [tf.reshape(ys, [-1], name='reshape_target')],
                [tf.ones([BATCH_SIZE * TIME_STEPS], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=ms_error,
                name='losses'
            )
        with tf.name_scope('average_cost'):
            cost = tf.div(
                    tf.reduce_sum(losses, name='losses_sum'),
                    BATCH_SIZE,
                    name='average_cost')
            tf.summary.scalar('cost', cost)
            
    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(LR).minimize(cost)
    result={'cost':cost,'train_op':train_op}
    return result
    
if __name__ == '__main__':
    model= LSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    optimizer = train_lstm(model['pred'])
    
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    plt.ion()
    plt.show()
    drawtrain=[]
    for i in range(250):
        seq, res = get_batch()
        feed_dict = {
                    xs: seq,
                    ys: res,
                    # create initial state
            }


        _, cost, state, pred = sess.run(
            [optimizer['train_op'],optimizer['cost'], model['final_state'], model['pred']],
            feed_dict=feed_dict)

#        if i % 5 == 0:
        print('cost: ', round(cost, 4))
        result = sess.run(merged, feed_dict)
        writer.add_summary(result, i)
        writer.flush()
        drawtrain.append(pred[:TIME_STEPS])
        
    
        
    drawpred = np.array(drawtrain).reshape([-1])
    pd_drawpred = pd.DataFrame(drawpred)
#    print(pd_drawpred[0])
    pd_or = pd.DataFrame(data)
    
    train_pic = go.Scatter(x=pd_drawpred.index,y=pd_drawpred[0])
    origin_pic = go.Scatter(x=pd_or.index,y = pd_or[0])
    r = [train_pic,origin_pic]
    fig = go.Figure(data=r)
    py.plot(fig)
    
    
    
    
    
    