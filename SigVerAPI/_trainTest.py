from verifier import *
import time
def trainAndTest(rate=0.0009, epochs=10000, neurons=7, display=False):    
    start = time()

    # Parameters
    global training_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs

    # Network Parameters
    n_hidden_1 = neurons # 1st layer number of neurons
    n_hidden_2 = 10 # 2nd layer number of neurons

    train_avg, test_avg = 0, 0
    n = 14
    for i in range(1,n+1):
        if display:
            print("Running for Person id",i)
        temp = ('0'+str(i))[-2:]
        train_score, test_score = evaluate(train_path.replace('01',temp), test_path.replace('01',temp))
        train_avg += train_score
        test_avg += test_score
    if display:
        print("Number of neurons in Hidden layer-", n_hidden_1)
        print("Number of neurons in Hidden layer-", n_hidden_2)
        print("Training average-", train_avg/n)
        print("Testing average-", test_avg/n)
        print("Time taken-", time()-start)
        print("Learning rate- ", learning_rate)
    return train_avg/n, test_avg/n, (time()-start)/n