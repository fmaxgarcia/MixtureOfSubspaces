import numpy as np

from scipy.optimize import minimize

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)



class MixtureOfSubspaces:

    def __init__(self, num_subspaces, proj_dimension, original_dimensions):
        '''
        subspaces: list of subspaces NxM, N samples, M dimensions
        '''
        #W experts weight matrix
        self.W = np.random.random( (num_subspaces, num_dimensions) )
        #g combination weight for predictors
        self.M = np.random.random( (num_subspaces, original_dimensions) )


    def _gating_weights(self, X, num_subspaces):
        ye = self.M.dot(X.T)
        g = np.zeros( (num_subspaces, X.shape[0]))
        for k in range(ye.shape[1]): 
            ge = softmax( ye[:,k] )
            g[:,k] = ge
        return g

    def _make_experts_prediction(self, X, X_proj):
        predictions = np.zeros( (len(X_proj), X.shape[0]))
        for i, x_proj in enumerate(X_proj):
            prediction = self.W[i].dot(x_proj.T)
            predictions[i] = prediction

        return predictions

    def make_prediction(self, X, X_proj):        
        predictions = self._make_experts_prediction(X, X_proj)
        g = self._gating_weights(X, len(X_proj))
        mixture_prediction = np.sum(g * predictions, axis=0)
        return mixture_prediction


    def _line_search(self, current_error, grad_w, grad_m, X, Y, X_proj):
        alpha = 1.0
        W_cpy = self.W.copy()
        M_cpy = self.M.copy()
        for i in range(5):
            self.W += alpha * grad_w
            self.M += alpha * grad_m
            new_error = np.mean( (Y - self.make_prediction(X, X_proj))**2, axis=0)
            self.W = W_cpy.copy()
            self.M = M_cpy.copy()
            if new_error < current_error:
                return alpha
            alpha /= 10
        return alpha

    def train_mixture(self, X, Y, X_proj):
        
        current_error = float("inf")
        while True:
            predictions = self._make_experts_prediction(X, X_proj)
            g = self._gating_weights(X, len(X_proj))
            mixture_prediction = np.sum(g * predictions, axis=0)

            MSE = np.mean( (Y - mixture_prediction)**2, axis=0)
            print "Error ", MSE 

            errors = Y - mixture_prediction
            grad_w = np.zeros( self.W.shape )
            grad_m = np.zeros( self.M.shape )
            for k in range(grad_w.shape[0]):
                weighted_error = (errors * g[k]).reshape((g[k].shape[0], 1))
                expert_error = (predictions[k] - mixture_prediction).reshape( (predictions[k].shape[0], 1) )

                grad_w[k] = np.mean(weighted_error * X_proj, axis=0)
                grad_m[k] = np.mean(weighted_error * expert_error * X, axis=0)

            alpha = self._line_search(MSE, grad_w, grad_m, X, Y, X_proj)
            self.W += alpha * grad_w
            self.M += alpha * grad_m

            if MSE - current_error > 0.1:
                print "Training finished..."
                break
            current_error = MSE
