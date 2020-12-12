# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    return np.loadtxt(filename)[:,:-1], np.loadtxt(filename)[:,-1]


# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    tr = int(np.round(train_proportion*len(x)))
    x_train = x[: tr]
    x_test = x[tr :]
    y_train = y[: tr]
    y_test = y[tr :]
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation, check our lecture slides
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
	inner = np.dot(x.T,x) + lambdaV*np.identity(len(x[0]))
	beta = np.dot(np.dot(np.linalg.inv(inner),x.T),y)
	return beta



# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    return np.sum(np.square(y-y_predict))/len(y)

# Given an array of x and theta predict y
def predict(x, theta):
    return np.dot(x,theta)

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train, lambdas):
	valid_losses = []
	training_losses = []
	for l in lambdas:
		intermediateLossesTraining = []
		intermediateLossesValidation = []
		for i in range(0,len(x_train)-3,4):
			trainingSetX = np.concatenate((x_train[:i,:],x_train[i+4:,:]))
			trainingSetY = np.concatenate((y_train[:i],y_train[i+4:]))
			validationSetX = x_train[i:i+4,:]
			validationSetY = y_train[i:i+4]
			beta = normal_equation(trainingSetX,trainingSetY,l)
			tl = get_loss(trainingSetY,predict(trainingSetX,beta))
			intermediateLossesTraining.append(tl)
			vl = get_loss(validationSetY,predict(validationSetX,beta))
			intermediateLossesValidation.append(vl)
		training_losses.append(np.mean(intermediateLossesTraining))
		valid_losses.append(np.mean(intermediateLossesValidation))

	return np.array(valid_losses), np.array(training_losses)



# Calcuate the l2 norm of a vector
def l2norm(vec):
	absvec = np.abs(vec)
	norm = np.dot(vec.T,vec)**(1/2)
	return norm

#  show the learnt values of β vector from the best λ
def bar_plot(beta):
	xvals = [i for i in range(len(beta))]
	plt.bar(xvals, beta)
	plt.title("Beta of Best Lambda --- Bar Chart")
	plt.show()


if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set(r"C:\Users\SMH\Documents\Nick\skule\Y2T1\CS4774\hw2\dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss")
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]


    # step 2: analysis
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = l2norm(normal_beta)# your code get l2 norm of normal_beta
    best_beta_norm = l2norm(best_beta)# your code get l2 norm of best_beta
    large_lambda_norm = l2norm(large_lambda_beta)# your code get l2 norm of large_lambda_beta
    print("Best Lambda:  " + str(best_lambda))
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))


    # step 3: visualization
    bar_plot(best_beta)
