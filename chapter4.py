# Fully connected neural networks applied to multiclass classification

import idx2numpy

# Downloaded all of these from kaggle, appendix j does not have
TRAIN_IMAGE_FILENAME = 'data/chapter4/train-images-idx3-ubyte'
TRAIN_LABEL_FILENAME = 'data/chapter4/train-labels-idx1-ubyte'
TEST_IMAGE_FILENAME = 'data/chapter4/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = 'data/chapter4/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

print('dimensions of training images:', train_images.shape)
print('dimensions of training labels:', train_labels.shape)
print('dimensions of test images:', test_images.shape)
print('dimensions of test labels:', test_labels.shape)

print('label for first training example:', train_labels[0])
print('--- pattern for first training example ---')
for line in train_images[0]:
    for num in line:
        if num > 0:
            print('*', end=' ')
        else:
            print(' ', end='')
    print('')
print('--- end pattern for first training example ---')

    