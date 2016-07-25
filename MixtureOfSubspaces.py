import numpy as np

from scipy.optimize import minimize
import math
from sys import stdout

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

CLASSIFICATION = 0
REGRESSION = 1

class MixtureOfSubspaces:

    def __init__(self, num_subspaces, proj_dimension, original_dimensions, num_outputs, task_type=REGRESSION):
        '''
        subspaces: list of subspaces NxM, N samples, M dimensions
        '''
        #W experts weight matrix
        self.W = np.random.random( (num_subspaces, num_outputs, proj_dimension) )
        #g combination weight for predictors
        self.M = np.random.random( (num_subspaces, num_outputs, original_dimensions) )
        self.num_outputs =num_outputs
        self.task_type = task_type


    def _gating_weights(self, X, num_subspaces):
        ye = self.M.dot(X.T)
        g = np.zeros( (num_subspaces, self.num_outputs, X.shape[0]))
        for i in range(ye.shape[1]):
            for k in range(ye.shape[2]):
                ge = softmax( ye[:,i,k] - np.max(ye[:,i,k], axis=0) )
                g[:,i, k] = ge
        return g

    def _make_experts_prediction(self, X, X_proj):
        predictions = np.zeros( (len(X_proj), self.num_outputs, X_proj[0].shape[0]))
        for i, x_proj in enumerate(X_proj):
            prediction = self.W[i].dot(x_proj.T)
            predictions[i] = prediction

        return predictions

    def make_prediction(self, X, X_proj):        
        predictions = self._make_experts_prediction(X, X_proj)
        g = self._gating_weights(X, len(X_proj))
        mixture_prediction = np.sum(g * predictions, axis=0)
        if self.task_type == CLASSIFICATION:
            mixture_prediction = mixture_prediction - np.max(mixture_prediction, axis=0)
            mixture_prediction = softmax(mixture_prediction)
        mixture_prediction = mixture_prediction.T

        return mixture_prediction


    def _line_search(self, current_error, grad_w, grad_m, X, Y, X_proj):
        alphas = [1.0] * self.num_outputs
        W_cpy = self.W.copy()
        M_cpy = self.M.copy()
        for i, a in enumerate(alphas):
            for j in range(5):
                self.W -= a * grad_w
                self.M -= a * grad_m
                prediction = self.make_prediction(X, X_proj)
                new_error = self._compute_error(Y, prediction)
                self.W = W_cpy.copy()
                self.M = M_cpy.copy()
                if new_error[i] < current_error[i]:
                    alphas[i] = a
                    break
                a /= 10
        return alphas

    def _compute_error(self, Y, prediction):
        if self.task_type == REGRESSION:
            return np.mean( (Y - prediction)**2, axis=0)
        elif self.task_type == CLASSIFICATION:
            return -np.log( np.sum( prediction[np.arange(Y.shape[0]), Y] ) )

    def estimate_gradient_W(self, X, X_proj, Y, current_error):
        print "estimating gradient for W"
        W_cpy = self.W.copy()
        step_size = 0.01
        w_grad = np.zeros( self.W.shape )
        count = 0
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                for k in range(self.W.shape[2]):
                    count += 1
                    stdout.write("\rW - %d/%d" %(count, self.W.shape[0]*self.W.shape[1]*self.W.shape[2]))
                    stdout.flush()
                    self.W = W_cpy.copy()
                    self.W[i,j,k] += step_size
                    prediction = self.make_prediction(X, X_proj)
                    new_error = self._compute_error(Y, prediction)
                    w_grad[i,j,k] = (new_error[j] - current_error[j]) / step_size

        return w_grad

    def estimate_gradient_M(self, X, X_proj, Y, current_error):
        print "estimating gradient for M"
        M_cpy = self.M.copy()
        step_size = 0.01
        m_grad = np.zeros( self.M.shape )
        count = 0
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                for k in range(self.M.shape[2]):
                    count += 1
                    stdout.write("\rM - %d/%d" %(count, self.M.shape[0]*self.M.shape[1]*self.M.shape[2]))
                    stdout.flush()
                    self.M = M_cpy.copy()
                    self.M[i,j,k] += step_size
                    prediction = self.make_prediction(X, X_proj)
                    new_error = self._compute_error(Y, prediction)
                    m_grad[i,j,k] = (new_error[j] - current_error[j]) / step_size

        return m_grad



    def train_mixture(self, X, Y, X_proj):
        
        current_error = float("inf")
        while True:
            predictions = self._make_experts_prediction(X, X_proj)
            g = self._gating_weights(X, len(X_proj))
            mixture_prediction = np.sum(g * predictions, axis=0)
            if self.task_type == CLASSIFICATION:
                mixture_prediction = mixture_prediction - np.max(mixture_prediction, axis=0)
                mixture_prediction = softmax(mixture_prediction)
            mixture_prediction = mixture_prediction.T

            loss = self._compute_error(Y, mixture_prediction)
            print "loss ", loss
            print "Sum loss ", np.sum(loss)

            grad_w = self.estimate_gradient_W(X, X_proj, Y, MSE)
            grad_m = self.estimate_gradient_M(X, X_proj, Y, MSE)
            # errors = Y - mixture_prediction
            # grad_w = np.zeros( self.W.shape )
            # grad_m = np.zeros( self.M.shape )
            # for k in range(grad_w.shape[0]):
            #     weighted_error = (errors * g[k].T).reshape( errors.shape )
            #     expert_error = (predictions[k].T - mixture_prediction).reshape( predictions[k].T.shape )
            #
            #     for l in range(self.W.shape[1]):
            #         reshape_error = weighted_error[:,l].reshape( (weighted_error.shape[0], 1))
            #         reshape_expert_error = expert_error[:,l].reshape( (expert_error.shape[0], 1) )
            #         grad_w[k,l] = -np.mean(reshape_error * X_proj[k], axis=0)
            #         grad_m[k,l] = -np.mean(reshape_error * reshape_expert_error * X, axis=0)

            alphas = self._line_search(loss, grad_w, grad_m, X, Y, X_proj)
            for i, alpha in enumerate(alphas):
                self.W[:,i,:] -= alpha * grad_w[:,i,:]
                self.M[:,i,:] -= alpha * grad_m[:,i,:]

            if math.fabs(np.sum(loss) - np.sum(current_error)) < 0.0001:
                print "Training finished..."
                break
            current_error = loss
