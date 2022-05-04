import json, zipfile
import numpy as np
from keras import models, layers

def r2_k(y_true, y_pred):
    #SS_res = K.sum(K.square(y_true - y_pred))
    #SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    #return (1 - SS_res / (SS_tot + epsilon))
    return 1.0 # Not required for inference

class k2rz():
    def __init__(self, model_path, n_models=1, ntheta=64, closed_surface=True, xpt_correction=True):
        self.nmodels = n_models
        self.ntheta = ntheta
        self.closed_surface = closed_surface
        self.xpt_correction = xpt_correction
        self.models = [models.load_model(model_path + f'/best_model{i}', custom_objects={'r2_k':r2_k}) for i in range(self.nmodels)]

    def set_inputs(self, ip, bt, βp, rin, rout, k, du, dl):
        self.x = np.array([ip, bt, βp, rin, rout, k, du, dl])

    def predict(self, post=True):
        #print('predicting...')
        self.y = np.zeros(2 * self.ntheta)
        for i in range(self.nmodels):
            self.y += self.models[i].predict(np.array([self.x]))[0] / self.nmodels
        rbdry, zbdry = self.y[:self.ntheta], self.y[self.ntheta:]
        if post:
            if self.xpt_correction:
                rgeo, amin = 0.5 * (max(rbdry) + min(rbdry)), 0.5 * (max(rbdry) - min(rbdry))
                if self.x[6] <= self.x[7]:
                    rx = rgeo - amin * self.x[7]
                    zx = max(zbdry) - 2 * self.x[5] * amin
                    rx2 = rgeo - amin * self.x[6]
                    rbdry[np.argmin(zbdry)] = rx
                    zbdry[np.argmin(zbdry)] = zx
                    rbdry[np.argmax(zbdry)] = rx2
                if self.x[6] > self.x[7]:
                    rx = rgeo - amin * self.x[6]
                    zx = min(zbdry) + 2 * self.x[5] * amin
                    rx2 = rgeo - amin * self.x[7]
                    rbdry[np.argmax(zbdry)] = rx
                    zbdry[np.argmax(zbdry)] = zx
                    rbdry[np.argmin(zbdry)] = rx2

            if self.closed_surface:
                rbdry = np.append(rbdry, rbdry[0])
                zbdry = np.append(zbdry, zbdry[0])

        return rbdry, zbdry

class kstar_lstm():
    def __init__(self, model_path, n_models=1):
        self.nmodels = n_models
        self.ymean = [1.30934765, 5.20082444, 1.47538417, 1.14439883]
        self.ystd  = [0.74135689, 1.44731883, 0.56747578, 0.23018484]
        self.models = []
        for i in range(self.nmodels):
            self.models.append(models.Sequential())
            self.models[i].add(layers.BatchNormalization(input_shape=(10, 21)))
            self.models[i].add(layers.LSTM(200, return_sequences=True))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.LSTM(200, return_sequences=False))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.Dense(200, activation='sigmoid'))
            self.models[i].add(layers.BatchNormalization())
            self.models[i].add(layers.Dense(4, activation='linear'))
            self.models[i].load_weights(model_path + f'/best_model{i}')

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 3 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.zeros_like(self.ymean)
        for i in range(self.nmodels):
            self.y += (self.models[i].predict(self.x)[0] * self.ystd + self.ymean) / self.nmodels
        return self.y

class kstar_nn():
    def __init__(self, model_path, n_models=1):
        self.nmodels = n_models
        self.ymean = [1.22379703, 5.2361062,  1.64438005, 1.12040048]
        self.ystd  = [0.72255576, 1.5622809,  0.96563557, 0.23868018]
        self.models = [models.load_model(model_path + f'/best_model{i}', custom_objects={'r2_k':r2_k}) for i in range(self.nmodels)]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.zeros_like(self.ymean)
        for i in range(self.nmodels):
            self.y += (self.models[i].predict(self.x)[0] * self.ystd + self.ymean) / self.nmodels
        return self.y

class bpw_nn():
    def __init__(self, model_path, n_models=1):
        self.nmodels = n_models
        self.ymean = np.array([1.02158800e+00, 1.87408512e+05])
        self.ystd  = np.array([6.43390272e-01, 1.22543529e+05])
        self.models = [models.load_model(model_path + f'/best_model{i}', custom_objects={'r2_k':r2_k}) for i in range(self.nmodels)]

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.zeros_like(self.ymean)
        for i in range(self.nmodels):
            self.y += (self.models[i].predict(self.x)[0] * self.ystd + self.ymean) / self.nmodels
        return self.y

class SB2_model():
    def __init__(self, model_path, low_state, high_state, low_action, high_action, activation='relu', last_actv='tanh', norm=True, bavg=0.):
        zf = zipfile.ZipFile(model_path)
        data = json.loads(zf.read('data').decode("utf-8"))
        self.parameter_list = json.loads(zf.read('parameter_list').decode("utf-8"))
        self.parameters = np.load(zf.open('parameters'))
        self.layers = data['policy_kwargs']['layers'] if 'layers' in data['policy_kwargs'].keys() else [64, 64]
        self.low_state, self.high_state = low_state, high_state
        self.low_action, self.high_action = low_action, high_action
        self.activation, self.last_actv = activation, last_actv
        self.norm = norm
        self.bavg = bavg

    def predict(self, x, yold=None):
        xnorm = 2 * (x - self.low_state) / (np.array(self.high_state) - self.low_state) - 1 if self.norm else x
        ynorm = xnorm
        for i,layer in enumerate(self.layers):
            w = self.parameters[f'model/pi/fc{i}/kernel:0']
            b = self.parameters[f'model/pi/fc{i}/bias:0']
            ynorm = np.matmul(ynorm,w) + b
            if self.activation == 'relu':
                ynorm = np.max([np.zeros_like(ynorm), ynorm], axis=0)
            elif self.activation == 'tanh':
                ynorm = np.tanh(ynorm)
            elif self.activation == 'sigmoid':
                ynorm = 1 / (1 + np.exp(-ynorm))
        w = self.parameters[f'model/pi/dense/kernel:0']
        b = self.parameters[f'model/pi/dense/bias:0']
        ynorm = np.matmul(ynorm,w) + b
        if self.last_actv == 'relu':
            ynorm = np.max([np.zeros_like(ynorm), ynorm], axis=0)
        elif self.last_actv == 'tanh':
            ynorm = np.tanh(ynorm)
        elif self.last_actv == 'sigmoid':
            ynorm = 1 / (1 + np.exp(-ynorm))

        y = 0.5 * (np.array(self.high_action) - self.low_action) * (ynorm + 1) + self.low_action if self.norm else ynorm
        if type(yold) == type(None):
            yold = x[:len(y)]
        y =  self.bavg * yold + (1 - self.bavg) * y
        return y

class SB2_ensemble():
    def __init__(self, model_list, low_state, high_state, low_action, high_action, activation='relu', last_actv='tanh', norm=True, bavg=0.):
        self.models = []
        for model_path in model_list:
            self.models.append(SB2_model(model_path, low_state, high_state, low_action, high_action, activation, last_actv, norm, bavg))

    def predict(self, x):
        ys = [m.predict(x) for m in self.models]
        return np.mean(ys, axis=0)


