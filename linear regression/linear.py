# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    return  np.loadtxt(filename)[:,:2], np.loadtxt(filename)[:,2]

# Find theta using the normal equation
def normal_equation(x, y):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)),np.transpose(x)),y)
    return theta

# Find thetas using stochastic gradient descent
def stochastic_gradient_descent(x, y, learning_rate, num_epoch):
    thetas = [[1,1]]
    for i in range(num_epoch):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        batch_x = x[randomize[0]]
        batch_y = y[randomize[0]]
        gradient = 2*np.dot(np.transpose(batch_x),(np.dot(batch_x,thetas[-1])-batch_y))/len(batch_x)
        thetas += [thetas[-1] - learning_rate*gradient]
    return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_epoch):
    thetas = [[1,1]]
    for i in range(num_epoch):
        gradient = 2*np.dot(np.transpose(x),(np.dot(x,thetas[-1])-y))/len(x)
        thetas += [thetas[-1] - learning_rate*gradient]
    return thetas

# Find thetas using minibatch gradient descent
def minibatch_gradient_descent(x, y, learning_rate, num_epoch, batch_size):
    thetas = [[1,1]]
    for i in range(num_epoch):
        randomize = np.arange(len(x))
        np.random.shuffle(randomize)
        batch_x = x[randomize[:batch_size]]
        batch_y = y[randomize[:batch_size]]
        gradient = 2*np.dot(np.transpose(batch_x),(np.dot(batch_x,thetas[-1])-batch_y))/len(batch_x)
        thetas += [thetas[-1] - learning_rate*gradient]
    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   return np.dot(x,theta)

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    return np.sum(np.square(y-y_predict))/len(y)

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_loss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:,1], y)
    plt.plot(x[:,1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # first column in data represents the intercept term, second is the x value, third column is y value
    x, y = load_data_set(r'C:\Users\SMH\Documents\Nick\skule\Y2T1\CS4774\regression-data.txt')
    # plot
    plt.scatter(x[:,1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

### Normal Equation ###
    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

### GD varying learning rate ###
    thetas = gradient_descent(x, y, 0.00001, 100) #low learning rate
    #plot(x, y, thetas[-1], "Gradient Descent Best Fit -- Low Learning Rate")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss -- Low Learning Rate (a=0.00001)")
    thetas = gradient_descent(x, y, 0.1, 100) #high learning rate
    #plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss -- High Learning Rate (a=0.1)")
    thetas = gradient_descent(x, y, 0.001, 100) #optimized learning rate
    plot(x, y, thetas[-1], "Gradient Descent Best Fit (a=0.001)")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss -- Optimal Learning Rate (a=0.001)")

#### SGD Varying learning rate ####
    #low learning_rate
    thetas = stochastic_gradient_descent(x, y, 0.01, 100)
    #plot(x, y, thetas[-1], "SGD Best Fit")
    plot_training_errors(x, y, thetas, "SGD Epoch vs Mean Training Loss -- Low Learning Rate (a=0.01)")
    thetas = stochastic_gradient_descent(x, y, 2, 100)
    #plot(x, y, thetas[-1], "SGD Best Fit")
    plot_training_errors(x, y, thetas, "SGD Epoch vs Mean Training Loss -- High Learning Rate (a=2)")
    thetas = stochastic_gradient_descent(x, y, 0.1, 100)
    plot(x, y, thetas[-1], "SGD Best Fit (a=0.1)")
    plot_training_errors(x, y, thetas, "SGD Epoch vs Mean Training Loss -- Optimal Learning Rate (a=0.1)")

### Mini-Batch GD, varying learning rate ###
    thetas = minibatch_gradient_descent(x, y, 0.00001, 100, 20)
    #plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch GD Epoch vs Mean Training Loss -- Low Learning Rate (a=0.00001)")
    thetas = minibatch_gradient_descent(x, y, 0.1, 100, 20)
    #plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch GD Epoch vs Mean Training Loss -- High Learning Rate (a=0.1)")
    thetas = minibatch_gradient_descent(x, y, 0.01, 100, 20)
    plot(x, y, thetas[-1], "Minibatch GD Best Fit (a=0.01)")
    plot_training_errors(x, y, thetas, "Minibatch GD Epoch vs Mean Training Loss -- Optimal Learning Rate (a=0.01)")

### Mini-Batch GD, varying learning batch size ###
    thetas = minibatch_gradient_descent(x, y, 0.01, 100, 5)
    #plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch GD Epoch vs Mean Training Loss-- Low Batch Size (b=5;a=0.01)")
    thetas = minibatch_gradient_descent(x, y, 0.01, 100, 40)
    plot(x, y, thetas[-1], "Minibatch GD Best Fit b=40 a=0.01")
    plot_training_errors(x, y, thetas, "Minibatch GD Epoch vs Mean Training Loss-- Optimal Batch Size (b=5;a=0.01)")
    thetas = minibatch_gradient_descent(x, y, 0.01, 100, 100)
    #plot(x, y, thetas[-1], "Minibatch GD Best Fit b=40")
    plot_training_errors(x, y, thetas, "Minibatch GD Epoch vs Mean Training Loss-- High Batch Size (b=100;a=0.01)")
