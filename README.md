# Euclidean-distance-in-TensorFlow

A flexible function in TensorFlow, to calculate the Euclidean distance between all row vectors in a tensor, the output is a 2D numpy array.

To clarify the fuction, we represent the input tensor as ***I*** with shape (***n, m***), and the output as ***O*** with shape (***n, n***), and ***i, j*** are both integer in the range ***0~n***. 

***O[i,j]*** is the Euclidean distance between ***i-th*** row and ***j-th*** row of ***I***.
