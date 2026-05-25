

import numpy as np
def linear_data(list_n, list_depn, dist='gaussian'):
    """
    Generate linear dependent data for multiple clients.

    Args:
        list_n (list): List of sample sizes for each client.
        list_depn (list): List of dependency for each client.
        dist (str): Distribution type ('Gaussian' or 'Uniform').
        noise (float): Standard deviation of Gaussian noise added to y.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        tuple: Tuple containing:
            - list_x (list): List of numpy arrays for x from each client.
            - list_y (list): List of numpy arrays for y from each client.
    """
    Data_X = []
    Data_Y = []
    for n, d in zip(list_n, list_depn):
        if dist == 'gaussian':
            x = np.random.randn(n).reshape(-1,1)
            y = d * x + np.random.randn(n).reshape(-1,1)
        elif dist == 'uniform':
            x = np.random.uniform(-2, 2, n).reshape(-1,1)
            y = d * x + np.random.uniform(-2, 2, n).reshape(-1,1)
        elif dist == 'laplace':
            x = np.random.laplace(n).reshape(-1,1)
            y = d * x + np.random.laplace(n).reshape(-1,1)

        Data_X.append(x)
        Data_Y.append(y)

    return Data_X, Data_Y


def Sinusoid(x, y, w):
    return 1 + np.sin(w*x)*np.sin(w*y)

def Sinusoid_Generator(n,w):
    i = 0
    output = np.zeros([n,2])
    while i < n:
        U = np.random.rand(1)
        V = np.random.rand(2)
        x0 = -np.pi + V[0]*2*np.pi
        x1 = -np.pi + V[1]*2*np.pi
        if U < 1/2 * Sinusoid(x0,x1,w):
            output[i, 0] = x0
            output[i, 1] = x1
            i = i + 1
    return output[:,0], output[:,1]


def frequency_data(list_n, list_w):
    """
    Generate frequency dependent data for multiple clients.

    Args:
        list_n (list): List of sample sizes for each client.
        list_w (list): List of frequency

    Returns:
        tuple: Tuple containing:
            - list_x (list): List of numpy arrays for x from each client.
            - list_y (list): List of numpy arrays for y from each client.
    """
    Data_X = []
    Data_Y = []
    for n, f in zip(list_n, list_w):
        output = Sinusoid_Generator(n, f)
        x = output[0].reshape(-1,1)
        y = output[1].reshape(-1,1)
            
        Data_X.append(x)
        Data_Y.append(y)
       

    return Data_X, Data_Y


def functional_data(list_n):
    """
    Generate functional dependent data for multiple clients.

    Args:
        list_n (list): List of sample sizes for each client.

    Returns:
        tuple: Tuple containing:
            - list_x (list): List of numpy arrays for x from each client.
            - list_y (list): List of numpy arrays for y from each client.
    """
    
    x1 = np.random.rand(list_n[0]).reshape(-1, 1)
    y1 = np.sin(x1) + np.random.randn(list_n[0]).reshape(-1, 1)
    x2 = np.random.rand(list_n[1]).reshape(-1, 1)
    y2 = np.cos(x2) + np.random.randn(list_n[1]).reshape(-1, 1)
    x3 = np.random.rand(list_n[2]).reshape(-1, 1)
    y3 = x3**2 + np.random.randn(list_n[2]).reshape(-1, 1)

    Data_X = [x1, x2, x3]
    Data_Y = [y1, y2, y3]

    return Data_X, Data_Y