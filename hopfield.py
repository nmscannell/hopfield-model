import numpy as np
import random
from matplotlib import pyplot as plt

p0 = np.array([1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1])
p1 = np.array([1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1])
p2 = np.array([-1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1])
p3 = np.array([1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
p4 = np.array([-1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1])
p5 = np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1])
p6 = np.array([-1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
p7 = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1])
p8 = np.array([1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1])
p9 = np.array([-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1])
d = 30


def initialize_weights(digits):
    w_init = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            sum = 0
            for k in range(len(digits)):
                sum += digits[k][i] * digits[k][j]
            w_init[i, j] = sum/d
    return w_init


def flip_pixels(x, n):
    y = np.copy(x)
    idx = random.sample(range(30), n)
    for i in range(n):
        y[idx[i]] *= -1
    return y


def update_input(x, w, asynchronous=False):
    x_prev = np.zeros(x.shape)
    k = 0
    while not np.array_equal(x_prev, x):
        k += 1
        x_prev = np.copy(x)
        if asynchronous:
            for i in range(d):
                idx = random.randrange(d)
                idx = i
                x[idx] = np.sign(np.dot(w[idx, :], x))
        else:
            x = np.sign(np.dot(w, x))
        if k > 100:
            break
    return x


def run_experiment(num_correct, r, w, num_iterations=10, all_digits=True): # correct is 3 x 3
    for n in range(1, 4):
        for i in range(num_iterations):
            x0 = flip_pixels(p0, n * 2)
            x1 = flip_pixels(p1, n * 2)
            x0 = update_input(x0, w)
            x1 = update_input(x1, w)
            if all_digits:
                x2 = flip_pixels(p2, n * 2)
                x3 = flip_pixels(p3, n * 2)
                x2 = update_input(x2, w)
                x3 = update_input(x3, w)
                x4 = flip_pixels(p4, n * 2)
                x5 = flip_pixels(p5, n * 2)
                x4 = update_input(x4, w)
                x5 = update_input(x5, w)
                x6 = flip_pixels(p6, n * 2)
                x7 = flip_pixels(p7, n * 2)
                x6 = update_input(x6, w)
                x7 = update_input(x7, w)
                x8 = flip_pixels(p8, n * 2)
                x9 = flip_pixels(p9, n * 2)
                x8 = update_input(x8, w)
                x9 = update_input(x9, w)

            if np.array_equal(x0, p0):
                num_correct[r][n - 1] += 1
            if np.array_equal(x1, p1):
                num_correct[r][n - 1] += 1

            if all_digits:
                if np.array_equal(x2, p2):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x3, p3):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x4, p4):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x5, p5):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x6, p6):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x7, p7):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x8, p8):
                    num_correct[r][n - 1] += 1
                if np.array_equal(x9, p9):
                    num_correct[r][n - 1] += 1


correct = np.zeros((3, 3))
w1 = initialize_weights([p0, p1])
w2 = initialize_weights([p0, p1, p2])
w3 = initialize_weights([p0, p1, p2, p3, p4, p5, p6])
run_experiment(correct, 0, w1)
run_experiment(correct, 1, w2)
run_experiment(correct, 2, w3)
print(correct)

correct = np.true_divide(correct, 100)
error = 1 - correct
print(correct)
num_stored = [2, 3, 7]
plt.plot(num_stored, correct[:, 0], 'r', num_stored, correct[:, 1], 'b-', num_stored, correct[:, 2], 'g')
plt.ylabel('Accuracy')
plt.xlabel('Number of Stored Digits')
plt.xticks(num_stored)
plt.show()
plt.plot(num_stored, error[:, 0], 'r', num_stored, error[:, 1], 'b-', num_stored, error[:, 2], 'g')
plt.ylabel('Error')
plt.xlabel('Number of Stored Digits')
plt.xticks(num_stored)
plt.show()
