import tensorflow as tf
import numpy as np

x = tf.constant([[1, 2, 3, 4],
                 [4, 5, 6, 7],
                 [7, 8, 9, 10]], tf.float32)


def euclidean_self(input_tensor):
    tensor_shape = input_tensor.get_shape().as_list()
    tensor_iter = input_tensor
    euclidean_list = []

    ses = tf.Session()

    for i in range(tensor_shape[0]):
        split_head, split_tail = tf.split(tensor_iter, [1, tensor_shape[0]-1])
        tensor_iter = tf.concat([split_tail, split_head], 0)
        euclidean_dist = (tf.sqrt(tf.reduce_sum(tf.square(input_tensor-tensor_iter), 1)))
        ses.run(euclidean_dist)
        euclidean_row = euclidean_dist.eval(session=ses)
        euclidean_list.append(euclidean_row)

    ses.close()

    euclidean_out = np.asarray(euclidean_list).transpose([1, 0])

    for i in range(tensor_shape[0]):
        euclidean_out[i] = np.append(euclidean_out[i][tensor_shape[0]-1-i:], 
                                     euclidean_out[i][:tensor_shape[0]-1-i])

    '''
    the return type is numpy.array, if need tenor for return, use the code beneath
    '''
    # euclidean_out = tf.convert_to_tensor(euclidean_out)

    return euclidean_out


print(euclidean_self(x))


