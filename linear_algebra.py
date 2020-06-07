#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List


# In[2]:

Vector = List[float]


height_weight_age = [70, 170, 40]


# In[3]:


grades = [95, 80, 75, 62]


# In[5]:


def add(v, w):
    assert len(v) == len(w)
    return [v_i + w_i for v_i, w_i in zip(v,w)]


# In[8]:


assert(add([1,2,3],[4,5,6]) == [5,7,9])


# In[9]:


def subtract(v, w):
    assert len(v) == len(w)
    return [v_i - w_i for v_i, w_i in zip(v,w)]


# In[12]:


assert(subtract([5,7,9],[4,5,6]) == [1,2,3])


# In[13]:


def vector_sum(vectors):
    assert vectors
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors)
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


# In[15]:


assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]


# In[20]:


def scalar_multiply(c, w):
    return [c*w_i for w_i in w]


# In[21]:


assert scalar_multiply(2,[3,5]) == [6,10]


# In[23]:


def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


# In[24]:


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


# In[25]:


def dot(v, w) -> float:
    assert len(v) == len(w)
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


# In[27]:


assert dot([1, 2, 3], [4, 5, 6]) == 32


# In[28]:


def sum_of_squares(v):
    return dot(v, v)


# In[29]:


assert sum_of_squares([1, 2, 3]) == 14 


# In[30]:


import math
def magnitude(v):
    return math.sqrt(sum_of_squares(v)) 


# In[31]:


assert magnitude([3, 4]) == 5


# In[33]:


def squared_distance(v, w):
    return sum_of_squares(subtract(v, w))


# In[34]:


def distance(v, w):
    return math.sqrt(squared_distance(v, w))


# In[35]:


def distance(v, w): 
    return magnitude(subtract(v, w))


# In[36]:


Matrix = List[List[float]]


# In[37]:


A = [[1, 2, 3], 
     [4, 5, 6]]
B = [[1, 2],    
     [3, 4],
     [5, 6]]


# In[38]:


from typing import Tuple


# In[39]:


def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 
    return num_rows, num_cols


# In[40]:


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)


# In[41]:


def get_row(A, i):
    return A[i]


# In[42]:


def get_column(A, j):
    return [A_i[j] for A_i in A]


# In[43]:


from typing import Callable


# In[44]:


def make_matrix(num_rows,num_cols,entry_fn):
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)] 


# In[45]:


def identity_matrix(n):
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


# In[46]:


assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]


# In[47]:


friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#            user 0  1  2  3  4  5  6  7  8  9
#


# In[48]:


friend_matrix = [[0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]


# In[49]:


assert friend_matrix[0][2] == 1, "0 and 2 are friends"
assert friend_matrix[0][8] == 0, "0 and 8 are not friends"
friends_of_five = [i
                   for i, is_friend in enumerate(friend_matrix[5])
                   if is_friend]


# In[ ]:




