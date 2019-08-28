import matplotlib.pyplot as plt
import csv



def mean_square_error2D(xVals, yVals, weight, bias):
    # cost function
    size = len(xVals)
    total_error = 0

    for i in range(size):
        total_error += (yVals[i] - (weight*xVals[i]) + bias)**2
    return total_error / size


def updateLine2D(xVals, yVals, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    size = len(xVals)

    for i in range(size):
        # weight partial derivative: -2x(y - (mx + b))
        weight_deriv += -2*xVals[i]*(yVals[i] - (weight*xVals[i] + bias))

        # bias partial derivative: -2(y - (mx + b))
        bias_deriv += -2*(yVals[i] - (weight*xVals[i] + bias))

    # subtract weights and biases to move in direction of steepest slope 'downwards'
    weight -= (weight_deriv / size) * learning_rate
    bias -= (bias_deriv / size) * learning_rate

    return weight, bias

def train2D(xVals, yVals, weight, bias, learning_rate, iterations):
    cost_history = []

    for i in range(iterations):
        weight, bias = updateLine2D(xVals, yVals, weight, bias, learning_rate)

        # calculate cost for auditing purposes
        cost = mean_square_error2D(xVals, yVals, weight, bias)
        cost_history.append(cost)

        # log progress
        if i % 1 == 0:
            print ("iter={:d}    weight={:.2f}    bias={:.4f}    cost={:.2}".format(i, weight, bias, cost))
    
    return weight, bias, cost_history

def plot(learning_rate, iterations):
    with open('train.csv', 'r') as f:
        data = list(csv.reader(f, delimiter=','))
        data.pop(0)


    xVals = []
    yVals = []
    for i in range(len(data)):
        xVals.append(float(data[i][0]))
        yVals.append(float(data[i][1]))


    xVals = [1,10]
    yVals = [2,10]

    train2D(xVals, yVals, 0.5, 0.5, learning_rate, iterations)

plot(0.001, 30)