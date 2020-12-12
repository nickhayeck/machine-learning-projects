# Machine Learning HW2 Poly Regression #

import matplotlib.pyplot as plt
import numpy as np

def plot(x, y):
    # plot
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    return np.loadtxt(filename)[:,0], np.loadtxt(filename)[:,1]

# Find theta using the normal equation
def normal_equation(x, y):
    theta = np.linalg.inv(x.T.dot(x)).dot(x.T.dot(y))
    return np.array(theta)


# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    # initialize theta as [1 1]
    theta = np.zeros(np.size(x, 1))
    thetas = []
    for i in range(num_iterations):
        loss = np.dot(x, theta) - y
        gradient = np.dot(x.T,loss)
        gradient /= len(x) # normalize by number of examples
        theta = theta - learning_rate*gradient
        thetas.append(theta)
    return np.array(thetas)

# Given an array of y and y_predict return loss
# y: an array of size n
# y_predict: an array of size n
# loss: a single float
def get_loss(y, y_predict):
    return np.sum(np.square(y-y_predict))/len(y)

# Given an array of x and theta predict y
# x: an array with size n x d
# theta: np array including parameters
# y_predict: prediction labels, an array with size n
def predict(x, theta):
    return np.dot(x,theta)


# Given a list of thetas one per (s)GD epoch
# this creates plots of epoch vs prediction loss (one about train, and another about validation or test)
# this figure checks GD optimization traits of the best theta
def plot_epoch_losses(x_train, x_test, y_train, y_test, best_thetas, title):
    losses = []
    tslosses = []
    epochs = []
    epoch_num = 1
    for theta in best_thetas:
        tslosses.append(get_loss(y_train, predict(x_train, theta)))
        losses.append(get_loss(y_test, predict(x_test, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses, label="training_loss")
    plt.plot(epochs, tslosses, label="testing_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()



# split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    tr = int(np.round(train_proportion*len(x)))
    x_train = x[: tr]
    x_test = x[tr :]
    y_train = y[: tr]
    y_test = y[tr :]
    return x_train, x_test, y_train, y_test


######################################################################
# Given a n by 1 dimensional array return an n by num_dimension array
# consisting of [1, x, x^2, ...] in each row
# x: input array with size n
# degree: degree number, an int
# result: polynomial basis based reformulation of x
######################################################################
def increase_poly_order(x, degree):
    result = []
    for elem in x:
        poly = []
        for j in range(degree+1):
            poly.append(elem**j)
        result.append(poly)
    return np.array(result)

# Give the parameter theta, best-fit degree , plot the polynomial curve
def best_fit_plot(x, y, theta, degree):
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

    #Calculates the x and y value for x evenly spaced 100 times between 0 and 10. Then uses these to plot the theta line
    x = np.linspace(-2,2,100)
    xp = increase_poly_order(x, degree)
    y_predict = predict(xp, theta)
    plt.plot(x, y_predict)
    plt.title("data vs fitted polynomial curve")
    plt.show()




##################################################################
# Given a list of degrees.
# For each degree in the list, train a polynomial regression.
# Return training loss and validation loss for a polynomial regression of order degree for
# each degree in degrees.
# Use 60% training data and 20% validation data. Leave the last 20% for testing later.
# Input:
# x: an array with size n x d
# y: an array with size n
# degrees: A list of degrees
# Output:
# training_losses: a list of losses on the training dataset
# validation_losses: a list of losses on the validation dataset
##################################################################

def get_loss_per_poly_order(x, y, degrees):
    #split data, leaving 20% for testing at the top of the dataset
    x_train, x_validate, y_train, y_validate = train_test_split(x,y,0.75)
    training_losses=[]
    validation_losses=[]
    for degree in degrees:
        degX = increase_poly_order(x_train, degree)
        valid_degX = increase_poly_order(x_validate, degree)
        weights = normal_equation(degX, y_train)
        training_losses.append(get_loss(y_train,predict(degX,weights)))
        validation_losses.append(get_loss(y_validate,predict(valid_degX,weights)))

    return training_losses, validation_losses


def select_hyperparameter(degrees, x_train, y_train):
    # Part 1: hyperparameter tuning:
    training_losses, validation_losses = get_loss_per_poly_order(x_train, y_train, degrees)
    plt.plot(degrees, training_losses, label="training_loss")
    plt.plot(degrees, validation_losses, label="validation_loss")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.title("poly order vs validation_loss")
    plt.show()

    # Part 2:  testing with the best learned theta
    valiFind = np.asarray(validation_losses)
    findd = np.where(valiFind == np.amin(valiFind))
    best_degree = degrees[int(findd[0])]
    print("Best Degree: " + str(best_degree))

    x_train_p = increase_poly_order(x_train, best_degree)
    best_theta = normal_equation(x_train_p, y_train)
    print("Best Theta: " +str(best_theta))
    return best_degree, best_theta


##################################################################
# Given a list of dataset sizes [d_1, d_2, d_3 .. d_k]
# Train a polynomial regression with first d_1, d_2, d_3, .. d_k samples
# Each time,
# return the a list of training and testing losses if we had that number of examples.
# We are using 0.5 as the training proportion because it makes the testing_loss more stable
# in reality we would use more of the data for training.
# Input:
# x: an array with size n x d
# y: an array with size n
# example_num: A list of dataset size
# Output:
# training_losses: a list of losses on the training dataset
# testing_losses: a list of losses on the testing dataset
#
# Given a list of sizes return the training and testing loss
# when using the given series number of examples.
##################################################################

def get_loss_per_num_examples(x, y, example_num, train_proportion):
    training_losses = []
    testing_losses = []
    for i in example_num:
        x_train, x_test, y_train, y_test = train_test_split(x[:i], y[:i], 0.5)
        theta = normal_equation(x_train, y_train)

        training_losses.append(get_loss(y_train, predict(x_train, theta)))
        testing_losses.append(get_loss(y_test, predict(x_test, theta)))
    return training_losses, testing_losses


if __name__ == "__main__":

    ##################################################################
    # Part 1: readin dataset / train , test split
    ##################################################################
    x, y = load_data_set(r"C:\Users\SMH\Documents\Nick\skule\Y2T1\CS4774\hw2\dataPoly.txt")
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    plot(x_train, y_train)
    plot(x_test, y_test)

    ########################################################
    # Part 2: select best_degree and derive its best_theta #
    ########################################################
    print("hyperparameter selection\n")
    degrees = [i for i in range(10)]
    best_degree, best_theta = select_hyperparameter(degrees, x_train, y_train)
    best_fit_plot(x_train, y_train, best_theta, best_degree)
    best_fit_plot(x_test, y_test, best_theta, best_degree)
    print("Best Theta:  " + str(best_theta))

    # model assessement to get the test loss
    x_testp = increase_poly_order(x_test, best_degree)
    print("Test Loss: "+str(get_loss(y_test, predict(x_testp, best_theta))))


    #############################################################################
    # Part 3: visual analysis to check GD optimization traits of the best theta #
    #############################################################################
    print("\n\n\nGradient Descent\n")
    print("Best Degree: "+str(best_degree))
    x_train_p = increase_poly_order(x_train, best_degree)
    x_test_p = increase_poly_order(x_test, best_degree)
    gbest_thetas = gradient_descent(x_train_p, y_train, 0.005, 1000)
    print("Best Theta: "+str(gbest_thetas[-1]))

    best_fit_plot(x_train, y_train, gbest_thetas[-1], best_degree)
    plot_epoch_losses(x_train_p, x_test_p, y_train, y_test, gbest_thetas, "best learned theta - train, test losses vs. GD epoch ")

    x_train_p = increase_poly_order(x_train, 3)
    x_test_p = increase_poly_order(x_test, 3)
    gbest_thetas = gradient_descent(x_train_p, y_train, 0.005, 1000)

    best_fit_plot(x_train, y_train, gbest_thetas[-1], 3)
    plot_epoch_losses(x_train_p, x_test_p, y_train, y_test, gbest_thetas,
                      "wrong theta - train, test losses vs. GD epoch (degree = 3)")


    ##################################################################
    # Part 4: analyze the effect of revising the size of train data:
    ##################################################################
    x, y = load_data_set(r"C:\Users\SMH\Documents\Nick\skule\Y2T1\CS4774\hw2\dataPoly.txt")
    x = increase_poly_order(x, 8)
    example_num = [10*i for i in range(1, 21)] # python list comprehension
    training_losses, testing_losses = get_loss_per_num_examples(x, y, example_num, 0.5)

    plt.plot(example_num, training_losses, label="training_loss")
    plt.plot(example_num, testing_losses, label="test_losses")
    plt.yscale("log")
    plt.legend(loc='best')
    plt.title("number of training examples vs training_loss and testing_loss")
    plt.show()
