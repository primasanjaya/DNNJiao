import tensorflow as tf
import random as rn
import numpy as np 
import os 
import argparse

import math 
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping 
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import pdb

'''
import skopt 
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer 
from skopt.plots import plot_convergence 
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args
'''

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report,precision_recall_fscore_support,accuracy_score
from scipy import stats

import sys

'''
dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')

dim_weight_decay = Real(low=1e-3, high = 0.5, prior = 'log-uniform', name='weight_decay')

dim_num_dense_layers = Integer(low=0, high = 5, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=5, high=1024, name='num_dense_nodes')

dim_activation = Categorical(categories=['relu', 'softplus'], name = 'activation')

dim_dropout = Real(low=1e-6, high=0.5, prior = 'log-uniform', name = 'dropout')

dimensions= [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes, dim_activation]

default_paramaters = [1e-4, 1e-3, 1e-6, 0, 100, 'relu']
'''

def log_dir_name(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation): 
    s = "./shuffled_crossvalidation%s_logs/lr_{0:.0e}_wd_{0:.0e}_layers_{2}_nodes{3}_{4}/"%fold
    log_dir = s.format(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
	#s = 'logs_{0:}/crossvalidation{1:}/lr_{2:.0e}_wd_{3:.0e}_layers_{4:}_nodes{5:}_{6:}'.format(dataset,fold,learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
    return log_dir 

### Make train test and validaiton here 
def create_model(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation): 
    ###Define model here 
    model = Sequential()
    model.add(InputLayer(input_shape = (input_size,)))
    for i in range(num_dense_layers): 
        name = 'layer_dense_{0}'.format(i+1)
        model.add(Dense(num_dense_nodes, activation=activation, name=name, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    #optimizer = Adam(lr=learning_rate)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    return model

'''
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation): 
	global best_accuracy
	#best_accuracy = 0.0
	print('learning rate: ',learning_rate)
	print('weight_decay: ', weight_decay)
	print('dropout', dropout)
	print('num_dense_layers: ', num_dense_layers)
	print('num_dense_nodes: ', num_dense_nodes)
	print('activation: ', activation)
	print() 
	model = create_model(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout, num_dense_layers=num_dense_layers, num_dense_nodes=num_dense_nodes, activation=activation)
	#log_dir = log_dir_name(learning_rate, weight_decay, num_dense_layers,num_dense_nodes, activation)
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	#history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=32, validation_data=validation_data,callbacks=[tensorboard_callback])
	history = model.fit(x=x_train, y=y_train, epochs=50, batch_size=32, validation_data=validation_data)
	accuracy = history.history['val_accuracy'][-1]
	print('Accuracy: {0:.2%}'.format(accuracy))
	if accuracy > best_accuracy: 
		model.save_weights(path_best_model)
		best_accuracy = accuracy 
	return -accuracy 
'''

def to_table(report):
    report = report.splitlines()
    res = []
    res.append(['']+report[0].split())
    for row in report[2:-2]:
        res.append(row.split())
    lr = report[-1].split()
    res.append([' '.join(lr[:3])]+lr[3:])
    return np.array(res)


def get_args():
	parser = argparse.ArgumentParser(description='Run DNN')
	parser.add_argument('--fold', type=int, default=None,
						help='fold')
	parser.add_argument('--input-file', type=str, default=None,
						help='input file')
	parser.add_argument('--model-dir', type=str, default=None,
						help='model dir')
	parser.add_argument('--output-file', type=str, default=None,
						help='output file')

	parser.add_argument('--pred', action='store_true', default=False)

	args = parser.parse_args()
	return args

if __name__ == '__main__': 

	args = get_args()
	path_best_model = args.model_dir + 'crossvalidation' + str(args.fold) + '_best_model.keras'
	data = pd.read_csv(args.input_file,index_col = 0)
	num_classes = 24

	#pdb.set_trace()

	x_test = data.iloc[:,:-1].values

	input_size = x_test.shape[1]

	if args.pred:
		hyps = np.load(args.model_dir + 'crossvalidation_results/fold' + str(args.fold) + '_hyperparams.npy')
		pd_hyps = pd.DataFrame(hyps)
		model = create_model(learning_rate=float(hyps[0][0]), weight_decay=float(hyps[0][1]), dropout=float(hyps[0][2]), num_dense_layers=int(hyps[0][3]), num_dense_nodes=int(hyps[0][4]), activation=hyps[0][5])
		model.load_weights(path_best_model)
	
	test_labels_names = ['Bone-Osteosarc',
						'Breast-AdenoCA',
						'CNS-GBM',
						'CNS-Medullo',
						'CNS-PiloAstro',
						'ColoRect-AdenoCA',
						'Eso-AdenoCA',
						'Head-SCC',
						'Kidney-ChRCC',
						'Kidney-RCC',
						'Liver-HCC',
						'Lung-AdenoCA',
						'Lung-SCC',
						'Lymph-BNHL',
						'Lymph-CLL',
						'Myeloid-MPN',
						'Ovary-AdenoCA',
						'Panc-AdenoCA',
						'Panc-Endocrine',
						'Prost-AdenoCA',
						'Skin-Melanoma',
						'Stomach-AdenoCA',
						'Thy-AdenoCA',
						'Uterus-AdenoCA']

	Y_pred = model.predict(x_test)

	pd_prob = pd.DataFrame(Y_pred)
	pd_prob.columns = test_labels_names

	y_pred = np.argmax(Y_pred, axis = 1)

	if y_pred.shape[0] == 1:
		pass
	else:
		y_pred = y_pred.squeeze()

	all_pred = []
	for i in y_pred:
		pred = test_labels_names[i]
		all_pred.append(pred)

	pd_prob['samples'] = data.iloc[:,-1]
	pd_prob['prediction'] = all_pred

	output_path = args.output_file.split('/')

	output_file = output_path[-1]
	output_dir = '/'.join(output_path[:-1]) + '/'
	pd_prob.to_csv(output_dir + 'model' + str(args.fold) + '_' + output_file)