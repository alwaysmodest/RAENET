"""
dataset.py
"""

# Necessary packages
import numpy as np
from scipy.special import expit
from sklearn.decomposition import PCA

def data_loading_twin(train_rate = 0.8):
    """Load twins data.

    Args:
    - train_rate: the ratio of training data

    Returns:
    - train_x: features in training data n*d
    - train_t: treatments in training data n*1
    - train_y: observed outcomes in training data n*1
    - train_potential_y: potential outcomes in training data n*2
    - test_x: features in testing data n*d
    - test_potential_y: potential outcomes in testing data    n*2
    """

    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    ori_data = np.loadtxt("data/Twin_data.csv", delimiter=",",skiprows=1,encoding = 'utf-8')

    # Define features
    x = ori_data[:,:30]
    no, dim = x.shape

    # Define potential outcomes
    potential_y = ori_data[:, 30:] # [2,n]
    # Die within 1 year = 1, otherwise = 0, 小于9999的代表1年内死亡，否则代表1年内存活
    potential_y = np.array(potential_y < 9999, dtype=float)

    ## Assign treatment
    coef = np.random.uniform(-0.01, 0.01, size = [dim,1])
    prob_temp = expit(np.matmul(x, coef) + np.random.normal(0,0.01, size = [no,1]))

    prob_t = prob_temp/(2*np.mean(prob_temp))
    prob_t[prob_t>1] = 1

    t = np.random.binomial(1,prob_t,[no,1])
    t = t.reshape([no,])

    ## Define observable outcomes
    y = np.zeros([no,1])
    y = np.transpose(t) * potential_y[:,1] + np.transpose(1-t) * potential_y[:,0]
    y = np.reshape(np.transpose(y), [no, ])

    ## Train/test division
    idx = np.random.permutation(no)
    train_idx = idx[:int(train_rate * no)]
    test_idx = idx[int(train_rate * no):]

    train_x = x[train_idx,:]
    train_t = t[train_idx]
    train_y = y[train_idx]
    train_potential_y = potential_y[train_idx,:]

    test_x = x[test_idx,:]
    test_t = t[test_idx]
    test_potential_y = potential_y[test_idx,:]

    return train_x, train_t, train_y, train_potential_y, test_x, test_t,test_potential_y

def data_loading_ihdp():
    data_in_train = np.load("data/ihdp_npci_1-100.train.npz")
    data_train = {'x': data_in_train['x'], 't': data_in_train['t'], 'yf': data_in_train['yf']}
    data_train['ycf'] = data_in_train['ycf']
    data_train['mu0'] = data_in_train['mu0']
    data_train['mu1'] = data_in_train['mu1']
    data_in_test = np.load("data/ihdp_npci_1-100.test.npz")
    data_test = {'x': data_in_test['x'], 't': data_in_test['t'], 'yf': data_in_test['yf']}
    data_test['ycf'] = data_in_test['ycf']
    data_test['mu0'] = data_in_test['mu0']
    data_test['mu1'] = data_in_test['mu1']
    return data_train['x'],data_train['t'],data_train['yf'],data_train['ycf'],data_train['mu0'],data_train['mu1'],data_test['x'],data_test['t'],data_test['yf'],data_test['ycf'],data_test['mu0'],data_test['mu1']
def data_twins():
    data_in_train = np.load("data/twins_1-10.train.npz")
    data_train = {'x': data_in_train['x'], 't': data_in_train['t'], 'yf': data_in_train['yf']}
    data_train['ycf'] = data_in_train['ycf']
    data_in_test = np.load("data/twins_1-10.test.npz")
    data_test = {'x': data_in_test['x'], 't': data_in_test['t'], 'yf': data_in_test['yf']}
    data_test['ycf'] = data_in_test['ycf']
    train_x=np.reshape(data_train['x'],(67200,25))
    train_t=np.reshape(data_train['t'],(67200,1))
    train_y=np.reshape(data_train['yf'],(67200,1))
    train_potential_y=np.reshape(data_train['ycf'],(67200,1))
    test_x=np.reshape(data_test['x'],(7500,25))
    test_t=np.reshape(data_test['t'],(7500,1))
    test_y=np.reshape(data_test['yf'],(7500,1))
    test_potential_y=np.reshape(data_test['ycf'],(7500,1))
    return train_x, train_t, train_y, train_potential_y, test_x, test_t, test_y, test_potential_y
def data_loading_jobs():
    data_in_train = np.load("data/jobs_DW_bin.new.10.test.npz")
    data_train = {'x': data_in_train['x'], 't': data_in_train['t'], 'yf': data_in_train['yf']}
    data_in_test = np.load("data/jobs_DW_bin.new.10.test.npz")
    data_test = {'x': data_in_test['x'], 't': data_in_test['t'], 'yf': data_in_test['yf']}
    data_test['e']=data_in_test['e']
    return data_train['x'], data_train['t'], data_train['yf'], data_test['x'], data_test['t'], data_test['yf'],data_in_test['e']
if __name__ == "__main__":
    train_x, train_t, train_y,train_potential_y,test_x, test_t,test_y,test_potential_y = data_loading_ihdp()
    # print(train_x.shape)
    # print(train_t.shape)
    # print(train_y.shape)
    # print(train_potential_y.shape)
    # print(test_x.shape)
    # print(test_t.shape)
    # print(test_y.shape)
    # print(test_potential_y.shape)
