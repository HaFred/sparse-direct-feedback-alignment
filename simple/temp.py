


import numpy as np
W = np.random.random((20, 2))

x = np.random.random((20,))

y_t = np.asarray([0.5, 0.5])

# for _ in xrange(10):
	
# 	y = np.dot(x, W)

# 	e = y_t - y

	

# 	dW = np.outer(x, e)

# 	W += dW*0.1


# 	print np.linalg.norm(e)



y = np.dot(x, W)
for _ in xrange(10):
	

	e = y_t - y

	y += 0.5*e
	
	print np.linalg.norm(e)
