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


def update_input(x, w):
    x_prev = np.zeros(x.shape)
    k = 0
    while not np.array_equal(x_prev, x):
        k += 1
        x_prev = np.copy(x)
        x = np.sign(np.dot(w, x))
        if k > 100:
            break
    return x


def run_experiment(w, num_flips=3, num_iterations=10, all_digits=True):
    result = np.zeros((len(w), num_flips))
    print(len(w))
    for row in range(len(w)):
        for n in range(1, num_flips+1):
            for i in range(num_iterations):
                x0 = flip_pixels(p0, n * 2)
                x1 = flip_pixels(p1, n * 2)
                if all_digits:
                    x2 = flip_pixels(p2, n * 2)
                    x3 = flip_pixels(p3, n * 2)
                    x4 = flip_pixels(p4, n * 2)
                    x5 = flip_pixels(p5, n * 2)
                    x6 = flip_pixels(p6, n * 2)
                    x7 = flip_pixels(p7, n * 2)
                    x8 = flip_pixels(p8, n * 2)
                    x9 = flip_pixels(p9, n * 2)
                x0 = update_input(x0, w[row])
                x1 = update_input(x1, w[row])
                if all_digits:
                    x2 = update_input(x2, w[row])
                    x3 = update_input(x3, w[row])
                    x4 = update_input(x4, w[row])
                    x5 = update_input(x5, w[row])
                    x6 = update_input(x6, w[row])
                    x7 = update_input(x7, w[row])
                    x8 = update_input(x8, w[row])
                    x9 = update_input(x9, w[row])

                if np.array_equal(x0, p0):
                    result[row][n - 1] += 1
                if np.array_equal(x1, p1):
                    result[row][n - 1] += 1
                if all_digits:
                    if np.array_equal(x2, p2):
                        result[row][n - 1] += 1
                    if np.array_equal(x3, p3):
                        result[row][n - 1] += 1
                    if np.array_equal(x4, p4):
                        result[row][n - 1] += 1
                    if np.array_equal(x5, p5):
                        result[row][n - 1] += 1
                    if np.array_equal(x6, p6):
                        result[row][n - 1] += 1
                    if np.array_equal(x7, p7):
                        result[row][n - 1] += 1
                    if np.array_equal(x8, p8):
                        result[row][n - 1] += 1
                    if np.array_equal(x9, p9):
                        result[row][n - 1] += 1
    return result


w1 = initialize_weights([p0, p1])
w2 = initialize_weights([p0, p1, p2])
w3 = initialize_weights([p0, p1, p2, p3, p4, p5, p6])
w4 = initialize_weights([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])
correct = run_experiment([w1, w2, w3, w4])
print(correct)

correct = np.true_divide(correct, 100)
error = 1 - correct
num_stored = [2, 3, 7, 10]
plt.plot(num_stored, correct[:, 0], 'r', label='2 flipped pixels')
plt.plot(num_stored, correct[:, 1], 'b', label='4 flipped pixels')
plt.plot(num_stored, correct[:, 2], 'g', label='6 flipped pixels')
plt.ylabel('Accuracy')
plt.xlabel('Number of Stored Digits')
plt.legend(loc='upper left')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_stored)
plt.savefig('hopfield_accuracy.png')

plt.figure()
plt.plot(num_stored, error[:, 0], 'r', label='2 flipped pixels')
plt.plot(num_stored, error[:, 1], 'b', label='4 flipped pixels')
plt.plot(num_stored, error[:, 2], 'g', label='6 flipped pixels')
plt.ylabel('Error')
plt.xlabel('Number of Stored Digits')
plt.legend(loc='bottom left')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_stored)
plt.savefig('hopfield_error.png')

correct = run_experiment([w1, w2, w3, w4], all_digits=False)
print(correct)

correct = np.true_divide(correct, 20)
error = 1 - correct
num_stored = [2, 3, 7, 10]
plt.figure()
plt.plot(num_stored, correct[:, 0], 'r', label='2 flipped pixels')
plt.plot(num_stored, correct[:, 1], 'b', label='4 flipped pixels')
plt.plot(num_stored, correct[:, 2], 'g', label='6 flipped pixels')
plt.ylabel('Accuracy')
plt.xlabel('Number of Stored Digits')
plt.legend(loc='bottom left')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_stored)
plt.savefig('hopfield_accuracy_2.png')

plt.figure()
plt.plot(num_stored, error[:, 0], 'r', label='2 flipped pixels')
plt.plot(num_stored, error[:, 1], 'b', label='4 flipped pixels')
plt.plot(num_stored, error[:, 2], 'g', label='6 flipped pixels')
plt.ylabel('Error')
plt.xlabel('Number of Stored Digits')
plt.legend(loc='upper left')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_stored)
plt.savefig('hopfield_error_2.png')

correct = run_experiment([w1, w2, w3, w4], num_flips=15)
print(correct)
correct = np.true_divide(correct, 100)
error = 1 - correct

plt.figure()
num_flipped = [2*i for i in range(1, 16)]
plt.plot(num_flipped, correct[0], 'r', label='2 stored vectors')
plt.plot(num_flipped, correct[1], 'b', label='3 stored vectors')
plt.plot(num_flipped, correct[2], 'g', label='7 stored vectors')
plt.plot(num_flipped, correct[3], 'k', label='10 stored vectors')
plt.ylabel('Accuracy')
plt.xlabel('Number of Flipped Pixels')
plt.legend(loc='upper right')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_flipped)
plt.savefig('flipped_accuracy.png')

plt.figure()
plt.plot(num_flipped, error[0], 'r', label='2 stored vectors')
plt.plot(num_flipped, error[1], 'b', label='4 stored vectors')
plt.plot(num_flipped, error[2], 'g', label='7 stored vectors')
plt.plot(num_flipped, error[3], 'k', label='10 stored vectors')
plt.ylabel('Error')
plt.xlabel('Number of Flipped Pixels')
plt.legend(loc='bottom right')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_flipped)
plt.savefig('flipped_error.png')

correct = run_experiment([w1, w2, w3, w4], num_flips=15, all_digits=False)
print(correct)
correct = np.true_divide(correct, 20)
error = 1 - correct

plt.figure()
num_flipped = [2*i for i in range(1, 16)]
plt.plot(num_flipped, correct[0], 'r', label='2 stored vectors')
plt.plot(num_flipped, correct[1], 'b', label='3 stored vectors')
plt.plot(num_flipped, correct[2], 'g', label='7 stored vectors')
plt.plot(num_flipped, correct[3], 'k', label='10 stored vectors')
plt.ylabel('Accuracy')
plt.xlabel('Number of Flipped Pixels')
plt.legend(loc='upper right')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_flipped)
plt.savefig('flipped_accuracy_2.png')

plt.figure()
plt.plot(num_flipped, error[0], 'r', label='2 stored vectors')
plt.plot(num_flipped, error[1], 'b', label='3 stored vectors')
plt.plot(num_flipped, error[2], 'g', label='7 stored vectors')
plt.plot(num_flipped, error[3], 'k', label='10 stored vectors')
plt.ylabel('Error')
plt.xlabel('Number of Flipped Pixels')
plt.legend(loc='bottom right')
plt.title('Performance of Hopfield Model on Simple Digits')
plt.xticks(num_flipped)
plt.savefig('flipped_error_2.png')