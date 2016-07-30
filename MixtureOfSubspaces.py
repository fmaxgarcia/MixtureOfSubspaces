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
            # mixture_prediction = mixture_prediction - np.max(mixture_prediction, axis=0)
            mixture_prediction = mixture_prediction / 1000.0
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
            return -np.log( np.sum( prediction[np.arange(Y.shape[0]), Y] ) )

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


    def _compute_gradient(self, g, Y, X, X_proj, mixture_prediction):
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
            for i in range( X_proj.shape[1]):
                stdout.write("\rInput - %d/%d" %(i, X_proj.shape[1]))
                stdout.flush()
                x = X_proj[:,i,:]
                index = Y[i]
                Wx = np.zeros( (self.W.shape[0], self.W.shape[1]))
                Mx = np.zeros( (self.M.shape[0], self.M.shape[1]))
                for e in range(x.shape[0]):
                    xe = x[e].reshape( (x.shape[1], 1))
                    out_w = self.W[e].dot(xe)
                    out_m = self.M[e].dot(X[i])
                    Wx[e] = out_w[:,0]
                    Mx[e] = out_m[:]
                Wx = Wx / 1000.0
                Mx = Mx / 1000.0
                Wx_g = Wx * g[:,:,i]
                e_Mx = np.exp( Mx )
                phi = np.exp( np.sum(Wx_g, axis=0) ).reshape( (self.num_outputs, 1))
                g_reshape = g[:,index,i].reshape( (g.shape[0], 1))

                grad_1w = x * g_reshape
                sum_phi_xg = np.zeros( grad_1w.shape )
                for j in range(phi.shape[0]):
                    sum_phi_xg += phi[j] * x * g[:,j,i].reshape( (g.shape[0], 1))
                grad_2w = (1 / np.sum(phi)) * sum_phi_xg
                grad_w[:,index,:] += -(grad_1w - grad_2w)

                grad_1m = np.zeros( (self.M.shape[0], self.M.shape[2]) )
                grad_2m = np.zeros( (self.M.shape[0], self.M.shape[2]) )
                sum_Wx = np.sum(Wx, axis=0)
                for j in range(self.M.shape[0]):
                    first_term = (Wx[j,:] - (sum_Wx[:] - Wx[j,:])).reshape((self.num_outputs, 1))
                    ### For numerical stability #####
                    # (np.prod(e_Mx, axis=0) / (np.sum(e_Mx, axis=0)**2)) same as log(np.prod(e_Mx, axis=0)) - 2 log(np.sum(e_Mx, axis=0))
                    #eprod_reshape = (np.prod(e_Mx, axis=0) / (np.sum(e_Mx, axis=0)**2)).reshape( (self.num_outputs, 1) )
                    numerator = np.sum(Mx, axis=0)
                    denominator = 2*np.log(np.sum(e_Mx, axis=0))
                    e_prod = np.exp(numerator - denominator).reshape( (self.num_outputs, 1) )
                    #################################
                    second_term = e_prod.dot( X[i].reshape( (X.shape[1], 1)).T )
                    grad_1m[j] = first_term[index] * second_term[index]
                    grad_2m[j] = (1 / np.sum(phi)) * ( np.sum(phi * first_term * second_term, axis=0) )

                grad_m[:,index,:] += -(grad_1m - grad_2m)
                if np.nonzero(np.isnan(grad_m))[0].shape[0] > 0:
                    a = 2

        return grad_w, grad_m





    def train_mixture(self, X, Y, X_proj):
        
        current_error = float("inf")
        while True:
            predictions = self._make_experts_prediction(X, X_proj)
            g = self._gating_weights(X, len(X_proj))
            mixture_prediction = np.sum(g * predictions, axis=0)
            if self.task_type == CLASSIFICATION:
                # mixture_prediction = mixture_prediction - np.max(mixture_prediction, axis=0)
                mixture_prediction = mixture_prediction / 1000.0
                mixture_prediction = softmax(mixture_prediction)
            mixture_prediction = mixture_prediction.T

            loss = self._compute_error(Y, mixture_prediction)
            print "loss ", loss
            print "Sum loss ", np.sum(loss)

            grad_w, grad_m = self._compute_gradient(g, Y, X, X_proj, mixture_prediction)
            # grad_w_est = self.estimate_gradient_W(X, X_proj, Y, loss)
            # grad_m_est = self.estimate_gradient_M(X, X_proj, Y, loss)

            alpha = self._line_search(loss, grad_w, grad_m, X, Y, X_proj)
            self.W -= alpha * grad_w
            self.M -= alpha * grad_m

            if math.fabs(np.sum(loss) - np.sum(current_error)) < 0.000001:
                print "Training finished..."
                break
            current_error = loss
