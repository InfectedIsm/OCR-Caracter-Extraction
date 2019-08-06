import tensorflow as tf
import numpy as np
import progressbar
import time

DEBUG = False

# =============================================================================================================================================
# =================================Useful functions=====================================================================================
# =============================================================================================================================================


# if the 'override' parameter is set to true, even if DEBUG=False
# the string is printed
def debugprint(s, override=False):
    if DEBUG or override:
        print(s)


def one_hot_decoding(vector, dimension, min_val=0):
    scalar = np.argmax(vector) + min_val
    return scalar


def new_batch(features, labels, batch_size):
    if type(features) == list and type(labels) == list:
        temp_features = np.zeros([batch_size, np.array(features[0]).shape[0]])
        temp_labels = np.zeros([batch_size, np.array(labels[0]).shape[0]])


    elif type(features) == np.ndarray and type(labels) == np.ndarray:
        temp_features = np.zeros([batch_size, features[0].shape[0]])
        temp_labels = np.zeros([batch_size, labels[0].shape[0]])

    else:
        raise AttributeError(
            'Error : accepted types : \'list\' or \'np.ndarray\' features and labels types must be the same')

    num_datas = len(features)
    random_indexes = np.random.randint(0, num_datas - 1, batch_size)

    for i, to_next_batch in enumerate(random_indexes):
        temp_features[i] = features[to_next_batch]
        temp_labels[i] = labels[to_next_batch]

    return temp_features, temp_labels
#=============================================================================================================================================
#=================================Neural Network Definition===================================================================================
#=============================================================================================================================================

class Model :
    def __init__(self, learning_rate, dropout_rate, io_size, hidden_layer_sizes):
        self.learning_rate = learning_rate
        self.dropout_rate = 1- dropout_rate
        self.io_size = io_size
        self.hidden_layer_sizes = hidden_layer_sizes


#============================================== 
#  Generate a layer, used in define_model() right below 
#============================================== 

def generate_layer(dimensions, input_tensor, last=False, dropout_rate=0.85):
    W = tf.Variable(tf.truncated_normal([dimensions[0],dimensions[1]], stddev=0.1),name='weights')
    b = tf.Variable(tf.zeros([dimensions[1]]),name='biais')
    if not last:    
        a_ = tf.nn.relu(tf.matmul(input_tensor,W)+b, name='output')
        a = tf.nn.dropout(a_, dropout_rate)
    else:
        a = tf.nn.softmax(tf.matmul(input_tensor,W)+b, name='output')
    return a

#============================================== 
#  This part generate the graph
#  For now, it dont take any parameters, but I aim to give it
# parameters and generate a model from them
#============================================== 

def define_model(model, debug=False):
    tf.reset_default_graph() 
    graph = tf.Graph()
    
    alpha = model.learning_rate
    dropout_rate = model.dropout_rate
    num_inputs = model.io_size[0]
    num_labels = model.io_size[1]
    layers = model.hidden_layer_sizes
    
    #create a vector with elements [num_inputs, layers, num_labels]
    all_layers = np.insert( np.insert( np.array(layers),len(layers),num_labels), 0, num_inputs) 
    if debug:    
        print(all_layers)
    
    #calculate the total number of parameters in the network
    all_layers_1 = all_layers + 1
    all_layers = np.insert(all_layers,len(all_layers),0)
    all_layers_1 = np.insert(all_layers_1,0,0)
    parameters_to_train = all_layers * all_layers_1
    parameters_to_train = np.sum(parameters_to_train)
    
    
    print('\n>> parameters to train : {}'.format(parameters_to_train))
    print('please wait ~1mn if no progress bar appear')
    
    layers.append(layers[len(layers)-1]) #needed to generate the last layer
        
    
    prev_layer_size = 0
    
    with graph.as_default():
        
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32, shape=[None,num_inputs],name='datas')
            y_ = tf.placeholder(tf.float32, shape=	[None,num_labels],name='labels')
            
        for layer_index,layer_size in enumerate(layers):
            layer_name = "layer-"+str(layer_index+1)
            if debug:            
                print(layer_name, end='')
    
            with tf.name_scope(layer_name):
                if layer_index == 0:
                    debugprint(" in"+str([num_inputs,layer_size]), override=debug)
                    layer_output=generate_layer([num_inputs,layer_size], x,dropout_rate=dropout_rate)
                
                elif layer_index == len(layers)-1:
                    debugprint(" out"+str([layer_size,num_labels]), override=debug)
                    y=generate_layer([prev_layer_size,num_labels], layer_output, last=True)
                    
                else:           
                    debugprint(" central"+str([prev_layer_size,layer_size]), override=debug)                
                    layer_output=generate_layer([prev_layer_size,layer_size], layer_output, dropout_rate=dropout_rate)
                    
            prev_layer_size = layer_size
            
        
        with tf.name_scope('categoricalCrossEntropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),
        						reduction_indices = [1]), name='crossEntropy')
                            
        with tf.name_scope('training'):
            alpha = 0.03
            train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

#        for op in tf.get_default_graph().get_operations():
#            print("["+str(op.name)+"]")      
    
        with tf.name_scope('prediction'):		
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')             
    
    return graph

#==============================================  
##  Just open a new session and return it
#==============================================

def open_session(graph):    
    
    with graph.as_default():    
#        all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
#        init = tf.variables_initializer(var_list=all_variables_list)
        init=tf.global_variables_initializer()
        
    session = tf.Session(graph=graph)
    session.run(init)
    tf.summary.FileWriter("tensorboard_test",session.graph)
    return session


#==============================================  
##  Train the network with as input a graph, and datas/labels
#==============================================


def training_run(graph, sess, x_train, y_train, number_of_training=1000,batch_size=100):
    with graph.as_default():
        
        loss_array = []
        smoothed_loss_array = []        
#        refresh_loss_rate = number_of_training*0.1
        train_step=tf.get_default_graph().get_operation_by_name("training/GradientDescent")
        accuracy=tf.get_default_graph().get_tensor_by_name("prediction/accuracy:0")    
        
        time.sleep(0.2) #without this pause, there is a bug with the progressbar
        with progressbar.ProgressBar(max_value=number_of_training) as bar:
            for actual_iter in range(number_of_training):
                bar.update(actual_iter)
                #create a batch of datas from train_datas vector
                x_batch, y_batch = new_batch(x_train, y_train, batch_size)                
                #get cross_entropy tensor from graph in order to use it
                cross_entropy = tf.get_default_graph().get_tensor_by_name("categoricalCrossEntropy/crossEntropy:0")
                                                
                _,loss,training_accuracy = sess.run([train_step,cross_entropy,accuracy], feed_dict={"inputs/datas:0": x_batch, "inputs/labels:0":y_batch})
                loss_array.append(loss)
                if len(loss_array)>20:
                    smoothed_loss_array.append(np.sum(loss_array[len(loss_array)-20:len(loss_array)])/20.)
                    
#                if not actual_iter % refresh_loss_rate :
##                    print("\nloss : {:.4f}\n ".format(loss))  
#                    plt.plot(loss_array)
#                    plt.title("loss")
#                    time.sleep(0.01)
#             plt.figure(2)
#             plt.plot(loss_array)
#             plt.plot(smoothed_loss_array, 'c')
#             plt.title("loss")
#             axes = plt.gca()
#             limit_low = min(loss_array)-min(loss_array)*0.1
#             limit_high = max(loss_array)+max(loss_array)*0.1
#             axes.set_ylim([limit_low,limit_high])
#             plt.show()
            print("last loss : {:.4f}".format(loss))
            print("training accuracy : {:.4f}".format(training_accuracy))
                        
    return loss,training_accuracy


#==============================================  
##  Test the network accuracy
#==============================================

def validation_run(graph, sess, x_test, y_test):
    with graph.as_default():    
        accuracy=tf.get_default_graph().get_tensor_by_name("prediction/accuracy:0")    
    print("\n>> accuracy is being calculated... please wait few seconds")
    validation_accuracy = sess.run(accuracy, feed_dict={"inputs/datas:0": x_test, "inputs/labels:0": y_test})
    print(">> model accuracy: ",validation_accuracy)    
    return 
