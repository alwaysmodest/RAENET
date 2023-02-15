import numpy as np
from torch.utils.data import Dataset,DataLoader
from data.data_loading import data_loading_twin,data_loading_ihdp,data_twins,data_loading_jobs

np.random.seed(1)

class twin_datasets(Dataset):
    def __init__(self, isTrain=True):
        super(twin_datasets, self).__init__()
        self.isTrain = isTrain
        train_x, train_t, train_y, train_potential_y, test_x,test_t,test_potential_y = data_loading_twin()

        if self.isTrain:
            self.x = train_x
            self.t = train_t
            self.y = train_y
            self.train_potential_y = train_potential_y
        else:
            self.x = test_x
            self.t = test_t
            self.test_potential_y = test_potential_y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.isTrain:
            return [self.x[index, :], self.t[index], self.y[index], self.train_potential_y[index]]
        else:
            return [self.x[index, :], self.t[index], self.test_potential_y[index]]

    @staticmethod
    def feature_dims():
        return 30
class twin_datasets_csv(Dataset):
    def __init__(self, isTrain=True):
        super(twin_datasets_csv, self).__init__()
        self.isTrain = isTrain
        train_x, train_t, train_y, train_potential_y, test_x,test_t,test_y,test_potential_y = data_twins()

        if self.isTrain:
            self.x = train_x
            self.t = train_t
            self.y = train_y
            self.train_potential_y = train_potential_y
        else:
            self.x = test_x
            self.t = test_t
            self.test_y=test_y
            self.test_potential_y = test_potential_y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.isTrain:
            return [self.x[index, :], self.t[index], self.y[index], self.train_potential_y[index]]
        else:
            return [self.x[index, :], self.t[index], self.test_y[index], self.test_potential_y[index]]

    @staticmethod
    def feature_dims():
        return 30
class ihdp_datasets(Dataset):
    def __init__(self, isTrain=True):
        super(ihdp_datasets, self).__init__()
        self.isTrain = isTrain
        train_x, train_t, train_y, train_potential_y,train_mu0,train_mu1,test_x,test_t,test_y,test_potential_y,test_mu0,test_mu1= data_loading_ihdp()

        if self.isTrain:
            self.x = train_x
            self.t = train_t
            self.y = train_y
            self.train_potential_y = train_potential_y
            self.train_mu0 = train_mu0
            self.train_mu1 = train_mu1
        else:
            self.x = test_x
            self.t = test_t
            self.test_y=test_y
            self.test_potential_y = test_potential_y
            self.test_mu0 = test_mu0
            self.test_mu1 = test_mu1
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, index):
        if self.isTrain:
            return [self.x[index, :], self.t[index], self.y[index], self.train_potential_y[index],self.train_mu0[index],self.train_mu1[index]]
        else:
            return [self.x[index, :], self.t[index], self.test_y[index], self.test_potential_y[index],self.test_mu0[index],self.test_mu1[index]]

    @staticmethod
    def feature_dims():
        return 30
class jobs_datasets(Dataset):
    def __init__(self, isTrain=True):
        super(jobs_datasets, self).__init__()
        self.isTrain = isTrain
        train_x, train_t, train_y,test_x,test_t,test_y,e= data_loading_jobs()

        if self.isTrain:
            self.x = train_x
            self.t = train_t
            self.y = train_y
        else:
            self.x = test_x
            self.t = test_t
            self.test_y=test_y
            self.e=e
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        if self.isTrain:
            return [self.x[index, :], self.t[index], self.y[index]]
        else:
            return [self.x[index, :], self.t[index], self.test_y[index],self.e[index]]

    @staticmethod
    def feature_dims():
        return 30
if __name__ == '__main__':
    train_dataset = twin_datasets(True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    for trainx, traint, trainy, trainpy in train_dataloader:
        print(trainx.shape)
        print(traint.shape)
        print(trainy.shape)
        print(trainpy.shape)
        break

    print("*"*30)

    test_dataset = twin_datasets(False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=True)
    for testx,testpy in test_dataloader:
        print(testx.shape)
        print(testpy.shape)
        break