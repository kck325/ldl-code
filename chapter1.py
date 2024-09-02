# The Rosenblatt Perceptron

def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += w[i] * x[i]
    if z < 0:
        return -1
    else:
        return 1
    
#print(compute_output([0.9, -0.6, -0.5], [1.0, -1.0, -1.0]))
#print(compute_output([0.9, -0.6, -0.5], [1.0, -1.0, 1.0]))
#print(compute_output([0.9, -0.6, -0.5], [1.0, 1.0, -1.0]))
#print(compute_output([0.9, -0.6, -0.5], [1.0, 1.0, 1.0]))

"""
Supervised learning - Ground truth and make the weights adjust to the data
Learning rate - hyperparameter, not adjusted by learning algorithm but can still be adjusted.
Learning rate will help with convergence.
Unsupervised learning - NLP
"""

import random

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], 'w1 =', '%5.2f' % w[1], 'w2 =', '%5.2f' % w[2])

random.seed(7)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

x_train = [
    [1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0]
]

y_train = [ 1.0, 1.0, 1.0, -1.0]

w = [0.2, -0.6, 0.25]

show_learning(w)

all_correct = False

while not all_correct:
    all_correct = True
    random.shuffle(index_list)
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x)
        if p_out != y:
            all_correct = False
            for j in range(len(w)):
                w[j] = w[j] + LEARNING_RATE * y * x[j]
            show_learning(w)

print('Final weights:')
show_learning(w)
    
# Skipping the matplotlib part of the chapter.
# A single perceptron can only classify linearly separable data.
# If the data is not linearly separable, we can use a multi-layer network.
# A multi-layer network can classify non-linearly separable data.
# Feedforward network - all connections are from left to right.
# Hidden layer - layer of neurons that are not visible.
# Output layer - layer of neurons that are visible.
# Cublas - CUDA BLAS - GPU BLAS - Runs linear algebra operations on the GPU.

import numpy as np

def compute_output(w, x):
    z = np.dot(w, x)
    return np.sign(z)

# Tensor - Extending matrix to another dimension makes it a 3D tensor
# Example of 4D tensor: Image of pixels with colors which are then organized into images.

