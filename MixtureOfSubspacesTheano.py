import numpy as np

import math
import theano
import theano.tensor as T
import lasagne

CLASSIFICATION = 0
REGRESSION = 1

class MixtureOfSubspacesTheano:

    def __init__(self, num_subspaces, proj_dimension, original_dimensions, num_outputs, learning_rate):
        self.learning_rate = learning_rate
        rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.expert_layers = []
        self.expert_inputs = []
    
        labels = T.ivector('labels')
        self.labels_shared = theano.shared( np.zeros((batch_size, 1), dtype='int32'), name='labels shared')
        self.expert_inputs = []
        self._get_outputs = []
        for i in range(num_subspaces):
            inputs_proj = T.matrix('input proj')
            inputs_proj_shared = theano.shared( np.zeros((batch_size, proj_dimension), dtype=theano.config.floatX))

            input_layer = lasagne.layers.InputLayer(shape=(batch_size, proj_dimension))
            hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=num_outputs, 
                            nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())
            
            output = lasagne.layers.get_output(hidden_layer, {input_layer : inputs_proj} )
            self.expert_inputs.append( inputs_proj_shared )
            self.expert_layers.append( hidden_layer )
            self.expert_outputs.append( output )

            self._get_outputs.append( theano.function([], [output], givens={inputs_proj : inputs_proj_shared}))


        self.gate_input = lasagne.layers.InputLayer(shape=(batch_size, original_dimensions))
        self.gate_layer = lasagne.layers.DenseLayer(self.gate_input, num_units=len(num_subspaces), 
                            nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.Uniform(), b=lasagne.init.Uniform())


        inputs_original = T.matrix('input original')
        self.inputs_original_shared = theano.shared( np.zeros((batch_size, original_dimensions), dtype=theano.config.floatX))
        gate_output = lasagne.layers.get_output(self.gate_layer, {self.gate_input : inputs_original}) 
        self._gate_outputs = theano.function([], [gate_output], givens={inputs_original: inputs_original})

        prediction = gate_output[0] * self.expert_outputs[0]
        for i in range(1, len(self.expert_outputs))
            predictions += gate_output[i] * self.expert_outputs[i]

        loss = T.mean( -T.log(T.max(predictions[T.arange(labels.shape[0]), labels])) )

        self._train = []
        for i in range(num_subspaces):
            expert_parameters = lasagne.layers.helper.get_all_params( self.expert_layers[i] )

            updates = lasagne.updates.sgd(error, self.all_parameters, self.learning_rate)
            self._train.append( theano.function([], [loss], updates=updates, givens={ inputs_proj : self.inputs_proj_shared}) )

        gate_parameters = lasagne.layers.helper.get_all_params( self.gate_layer )
        updates = lasagne.updates.sgd(error, gate_parameters, self.learning_rate)

        self._train.append( theano.function([], [loss], updates=updates, givens={inputs_original : self.inputs_original_shared }) )



    def train_mixture(self, X, Y, X_proj):
        for i in range(len(X_proj)):
            self.expert_inputs[i].set_value( X_proj[i] )
        self.inputs_original_shared.set_value( X )
        self.labels_shared.set_value( Y )

        for train in self._train:
            train()

    def make_prediction(self, X, X_proj):
        expert_outputs = []
        for i in range(len(X_proj)):
            self.expert_inputs[i].set_value( X_proj[i] )
            expert_outputs.append( self._get_outputs()[0] )

        self.inputs_original_shared.set_value( X )
        weights = self._get_outputs()[0]
        print "weights ", weights
        print "outputs ", expert_outputs


