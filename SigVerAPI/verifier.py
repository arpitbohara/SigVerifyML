import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from time import time
import keras
# from helperfunctions import *
from SigVerAPI.fileOperations import getCSVFeatures
import os
import warnings

tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore', category=FutureWarning)


from dotenv import load_dotenv
load_dotenv()
# genuine_image_paths = "D:\\Arpit College\\FYP\\SignatureVerificationSystem\\real/"
# FORGED_IMAGE_PATH = "D:\\Arpit College\\FYP\\SignatureVerificationSystem\\forged/"


def testing(path):
    feature = getCSVFeatures(path)
    test_csv_path=os.getenv('TEST_CSV_PATH')
    test_feature_folder=os.getenv('TEST_FEATURE_FOLDER')
    if not(os.path.exists(test_feature_folder)):
        os.mkdir(test_feature_folder)
    with open(test_csv_path, 'w') as handle:
        handle.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        handle.write(','.join(map(str, feature))+'\n')

def readCSV(train_path, test_path, type2=False):
    n_input = 9
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = keras.utils.to_categorical(correct,2)      # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not(type2):
        df = pd.read_csv(test_path, usecols=range(n_input))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = keras.utils.to_categorical(correct,2)      # Converting to one hot
    if not(type2):
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input

        




def evaluate(train_path, test_path, type2=False): 
    forged_image_paths=os.getenv("FORGED_IMAGE_PATH")
    genuine_image_paths=os.getenv("GENIUNE_IMAGE_PATH")

    # makeCSV(genuine_image_paths=genuine_image_paths,forged_image_paths=forged_image_paths)

    n_input = 9


    tf.reset_default_graph()
    # Parameters
    learning_rate = 0.0009
    training_epochs = 10000
    display_step = 1

    # Network Parameters
    n_hidden_1 = 7 # 1st layer number of neurons
    n_hidden_2 = 13 # 2nd layer number of neurons
    # n_hidden_3 = 13 # 3rd layer
    n_classes = 2 # no. of classes (genuine or forged)

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=2)),
        # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], seed=1)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], seed=2))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], seed=2)),
        # 'b3': tf.Variable(tf.random_normal([n_hidden_3], seed=2)),
        'out': tf.Variable(tf.random_normal([n_classes], seed=1))
    }


    # Create model
    def multilayer_perceptron(x):
        layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        out_layer = tf.tanh(tf.matmul(layer_2, weights['out']) + biases['out'])
        return out_layer

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer

    loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # For accuracies
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Initializing the variables
    init = tf.global_variables_initializer()  

    if not(type2):
        train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(train_path, test_path, type2)
    ans = 'Random'
    # costs=[]

    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):

            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})

            # if epoch==training_epochs-1:
            #     print (epoch , cost)

            # if epoch%100==0:
            #     print(epoch , cost)
            # elif epoch==training_epochs-1:
            #     print(epoch , cost)

            if cost<0.0001:
                break
#             # Display logs per epoch step
        #     if epoch % 999 == 0:
        #         print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
        print("Optimization Finished!")

         # Plot the cost values
        # plt.plot(costs)
        # plt.xlabel('Epoch')
        # plt.ylabel('Cost')
        # plt.title('Cost vs. Epoch')
        # plt.show()


        # Finding accuracies
        accuracy1 =  accuracy.eval({X: train_input, Y: corr_train})
        print("Accuracy for train:", accuracy1)
        
        if type2 is False:
            accuracy2 =  accuracy.eval({X: test_input, Y: corr_test})   
            print("Accuracy for test:", accuracy2)

            return accuracy1, accuracy2
            
        else:
            prediction = pred.eval({X: test_input})
            if prediction[0][1]>prediction[0][0]:
                print('Genuine Image')
                return True
            else:
                print('Forged Image')
                return False


# trainAndTest(display=True)
# train_person_id:str="001",test_image_path:str=r"D:\Arpit College\FYP\SignatureVerificationSystem\uploads\021001_000.png"
def verifySignature(train_person_id:str,test_image_path:str):
    # train_person_id = "001"
    # test_image_path = input("Enter path of signature image : ")
    train_base_folder=str(os.getenv('TRAINING_FEATURE_FOLDER'))
    train_path = train_base_folder+'/training_'+train_person_id+'.csv'
    print(train_path)
    testing(test_image_path)
    test_path = os.getenv('TEST_CSV_PATH')
    return evaluate(train_path, test_path, type2=True)