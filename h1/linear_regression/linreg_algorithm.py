def linear_regression(x, y) :
    
    if (x.ndim == 1) : x = scp.matrix(x).T
    else : x = scp.matrix(x)
    y = scp.matrix(y).T
    
    
    ones_column = scp.tile([1], (x.shape[0], 1))
    x = scp.concatenate((x, ones_column), axis=1)
    
    w = (((((x.T * x).I) * x.T))* y ).A
    
    b = w[-1,0]
    w = w[0:-1].flatten()
    
    predict = lambda new : scp.dot(w, new) + b
    
    return {'hyp' : predict, 'est_params' : {'weigths': w, 'bias': b}}