import numpy as np

from scipy.optimize import minimize
from scipy import optimize
import cma

import math
from sys import stdout

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

CLASSIFICATION = 0
REGRESSION = 1

class MixtureOfSubspaces:

    def __init__(self, num_subspaces, proj_dimension, original_training, num_outputs, task_type=REGRESSION):
        '''
        subspaces: list of subspaces NxM, N samples, M dimensions
        '''
        #W experts weight matrix
        self.W = np.random.random( (num_subspaces, num_outputs, proj_dimension) )
        #g combination weight for predictors
        self.M = np.zeros( (num_subspaces, original_training.shape[1]) )
        maxs = np.max(original_training, axis=0)
        mins = np.min(original_training, axis=0)
        step = (maxs - mins) / num_subspaces
        for i in range(num_subspaces):
            self.M[i,:] = (mins + (i*step)) - (step/2.0)

        self.num_outputs = num_outputs
        self.task_type = task_type


    def _gating_weights(self, X, num_subspaces):
        ye = self.M.dot(X.T)
        g = np.zeros( (num_subspaces, X.shape[0]))
        for k in range(ye.shape[1]):
            ge = softmax( ye[:,k] - np.max(ye[:,k], axis=0) )
            g[:,k] = ge
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
        mixture_prediction = np.zeros( predictions.shape )
        for i in range(predictions.shape[1]):
            mixture_prediction[:,i,:] = (predictions[:,i,:] * g)
        mixture_prediction = np.sum(mixture_prediction, axis=0)
        if self.task_type == CLASSIFICATION:
            mixture_prediction = softmax(mixture_prediction)
        mixture_prediction = mixture_prediction.T

        return mixture_prediction


    def _line_search(self, current_error, grad_w, grad_m, X, Y, X_proj):
        a = 1.0
        W_cpy = self.W.copy()
        M_cpy = self.M.copy()
        for j in range(20):
            self.W -= a * grad_w
            self.M -= a * grad_m
            prediction = self.make_prediction(X, X_proj)
            new_error = self._compute_error(Y, prediction)
            self.W = W_cpy.copy()
            self.M = M_cpy.copy()
            if isinstance(current_error, float):
                if new_error < current_error:
                    print "Alpha ", a
                    break
            else:
                if new_error[i] < current_error[i]:
                    break
            a /= 10
        return a

    def _compute_error(self, Y, prediction):
        if self.task_type == REGRESSION:
            return np.mean( (Y - prediction)**2, axis=0)
        elif self.task_type == CLASSIFICATION:
            return -np.sum( np.log(prediction[np.arange(Y.shape[0]), Y] ) )

    def estimate_gradient_W(self, X, X_proj, Y, current_error):
        print "estimating gradient for W"
        step_size = 0.01
        w_grad = np.zeros( self.W.shape )
        count = 0
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                for k in range(self.W.shape[2]):
                    count += 1
                    stdout.write("\rW - %d/%d" %(count, self.W.shape[0]*self.W.shape[1]*self.W.shape[2]))
                    stdout.flush()
                    self.W[i,j,k] += step_size
                    prediction = self.make_prediction(X, X_proj)
                    new_error = self._compute_error(Y, prediction)
                    self.W[i,j,k] -= step_size
                    if isinstance(current_error, float):
                        w_grad[i,j,k] = (new_error - current_error) / step_size
                    else:
                        w_grad[i,j,k] = (new_error[j] - current_error[j]) / step_size

        return w_grad

    def estimate_gradient_M(self, X, X_proj, Y, current_error):
        print "estimating gradient for M"
        step_size = 0.01
        m_grad = np.zeros( self.M.shape )
        count = 0
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                for k in range(self.M.shape[2]):
                    count += 1
                    stdout.write("\rM - %d/%d" %(count, self.M.shape[0]*self.M.shape[1]*self.M.shape[2]))
                    stdout.flush()
                    self.M[i,j,k] += step_size
                    prediction = self.make_prediction(X, X_proj)
                    new_error = self._compute_error(Y, prediction)
                    self.M[i,j,k] -= step_size
                    if isinstance(current_error, float):
                        m_grad[i,j,k] = (new_error - current_error) / step_size
                    else:
                        m_grad[i,j,k] = (new_error[j] - current_error[j]) / step_size

        return m_grad


    def _compute_gradient(self, g, Y, X, X_proj, mixture_prediction, predictions):
        grad_w = np.zeros( self.W.shape )
        grad_m = np.zeros( self.M.shape )
        if self.task_type == REGRESSION:
            errors = Y - mixture_prediction
            for k in range(grad_w.shape[0]):
                weighted_error = (errors * g[k].T).reshape( errors.shape )
                expert_error = (predictions[k].T - mixture_prediction).reshape( predictions[k].T.shape )

                for l in range(self.W.shape[1]):
                    reshape_error = weighted_error[:,l].reshape( (weighted_error.shape[0], 1))
                    reshape_expert_error = expert_error[:,l].reshape( (expert_error.shape[0], 1) )
                    grad_w[k,l] = -np.mean(reshape_error * X_proj[k], axis=0)
                    grad_m[k,l] = -np.mean(reshape_error * reshape_expert_error * X, axis=0)
        else:
            X_proj = np.asarray(X_proj)
            for i in range(X_proj.shape[1]):
                stdout.write("\rInput - %d/%d" %(i, X_proj.shape[1]))
                stdout.flush()
                correct_index = Y[i]
                for k in range(self.W.shape[0]):
                    sum_pred = 0.0
                    for j in range(self.num_outputs):
                        sum_pred += self.W[k, j].dot( X_proj[k, i] ) * mixture_prediction[i, j]

                        if correct_index == j:
                            grad_w[k,j] -= g[k,i] * X_proj[k,i] * (1 - mixture_prediction[i,correct_index])
                        else:
                            grad_w[k,j] -= (- X_proj[k,i] * g[k,i] * mixture_prediction[i,j])
                    if correct_index == j:
                        grad_m[k] -= ( self.W[k, correct_index].dot( X_proj[k,i] ) - sum_pred) * (X_proj[k,i] * g[k,i] * (1 - g[k,i]))
                    else:
                        grad_m[k] -=  ( self.W[k, correct_index].dot( X_proj[k,i] ) - sum_pred) * (-g[correct_index, i] * X_proj[k,i] * g[k, i])

        if len(np.nonzero(np.isnan(grad_m))[0]) > 0 or len(np.nonzero(np.isnan(grad_w))[0]) > 0:
            print np.nonzero(np.isnan(grad_m))
            print np.nonzero(np.isnan(grad_w))
            print "Nan grad"
            assert False
        return grad_w, grad_m



    def error_function(self, x, *args):
        X, Y, X_proj = args
        w_params = np.asarray( x[:self.W.shape[0]*self.W.shape[1]*self.W.shape[2]] )
        m_params = np.asarray( x[self.W.shape[0]*self.W.shape[1]*self.W.shape[2]:] )
        self.W = w_params.reshape( self.W.shape )
        self.M = m_params.reshape( self.M.shape )
        prediction = self.make_prediction(X, X_proj)
        error = self._compute_error(Y, prediction)
        print "Error ", error
        return  error

    def train_mixture(self, X, Y, X_proj):

        # params = self.W.flatten()
        # params = np.hstack( (params, self.M.flatten()))
        # # result = cma.fmin(objective_function=self.error_function, x0=params.tolist(), sigma0=1.0, options={'maxiter':20}, args=(X,Y,X_proj))
        # result = optimize.fmin_bfgs(f=self.error_function, x0=[ params ], epsilon=0.1, args=(X,Y,X_proj))

        current_error = float("inf")
        step = 0
        while True:
            predictions = self._make_experts_prediction(X, X_proj)
            g = self._gating_weights(X, len(X_proj))
            mixture_prediction = np.zeros( predictions.shape )
            for i in range(predictions.shape[1]):
                mixture_prediction[:,i,:] = (predictions[:,i,:] * g)
            mixture_prediction = np.sum(mixture_prediction, axis=0)
            if self.task_type == CLASSIFICATION:
                mixture_prediction = softmax(mixture_prediction)
            mixture_prediction = mixture_prediction.T

            loss = self._compute_error(Y, mixture_prediction)
            print "loss ", loss
            print "Sum loss ", np.sum(loss)

            grad_w, grad_m = self._compute_gradient(g, Y, X, X_proj, mixture_prediction, predictions)

            # grad_w_est = self.estimate_gradient_W(X, X_proj, Y, loss)
            # grad_m_est = self.estimate_gradient_M(X, X_proj, Y, loss)

            alpha = self._line_search(loss, grad_w, grad_m, X, Y, X_proj)
            self.W -= alpha * grad_w
            self.M -= alpha * grad_m
            step += 1

            if math.fabs(np.sum(loss) - np.sum(current_error)) < 0.00000001:
                print "Training finished..."
                break
            current_error = loss
