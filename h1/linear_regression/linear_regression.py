import pandas as pd
import scipy as scp
import matplotlib.pyplot as plt

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

def make_plot(x, y, models, name) : 
    
    fig, ax = plt.subplots(1, 1)
    
    ax.scatter(x,y)
    
    xs = scp.arange(x.min() - 0.1, x.max() + 0.1, 0.1)
    
    for model in models :   
        ys = scp.vectorize(model)(xs)
        ax.plot(xs, ys)
    
    ax.set_xlabel('Temperatur', fontsize = 'large')
    ax.set_ylabel('Energy', fontsize = 'large')
    
    fig.savefig(name)
    
    return

def mse(x, y, model) :
    
    pred = scp.vectorize(model)(x)
    errors = y - pred
    
    return scp.sum(errors ** 2) / errors.shape[0]

def var(y) :
    return scp.sum((y - scp.mean(y)) ** 2) / y.shape[0]

danwood = pd.read_table('DanWood.dt', header = None, sep = ' ')

danwood.columns = ['temp','energy']

x = danwood.temp.values

y = danwood.energy.values

reg = linear_regression(x,y)

model = reg['hyp']

reg['est_params']['weigths']

reg['est_params']['bias']

var(y)

mse(x,y,model)

mse(x, y, model) / var(y)

make_plot(x, y, [model], 'reg_plot_1.jpg')

constant_model = lambda x : scp.mean(y)

make_plot(x, y, [model, constant_model], 'reg_plot_2.jpg')

