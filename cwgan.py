import tensorflow as tf
import math
import numpy as np

class Dense(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
     super(Dense, self).__init__()
     bound = 1/math.sqrt(input_dim)
     w_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
     self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
     b_init = tf.keras.initializers.RandomUniform(minval=-bound, maxval=bound)
     self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )
  def call(self,inputs):
      return tf.matmul(inputs, self.w) + self.b


class Cross(tf.keras.layers.Layer):
       """
        Cross Layer, see Wang, Fu, Fu and Wang (2017): Deep & Cross Network for Ad Click Predictions
       """
       def __init__(self, input_features):
            super(Cross, self).__init__()
            self.input_features = input_features
            w_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2/input_features))
            self.w = tf.Variable(
             initial_value=w_init(shape=(self.input_features,), dtype="float32"),
             trainable=True, name='weights'
            )
            b_init = tf.keras.initializers.Constant(0.1)
            self.b = tf.Variable(
                initial_value=b_init(shape=(self.input_features,), dtype="float32"), trainable=True, name='bias'
            )

       def call(self, x0, x_cross):
            x0xl = tf.matmul(tf.expand_dims(x0, -1), tf.expand_dims(x_cross, -2))
            out = tf.tensordot(x0xl, self.w, [[-1], [0]]) + self.b + x_cross
            return out

def get_activation_by_name(name: str):
       activation_dict = {'relu': tf.keras.activations.relu,
                       'leaky_relu': tf.keras.layers.LeakyReLU(),
                       'tanh': tf.keras.activations.tanh,
                       'sigmoid': tf.keras.activations.sigmoid,
                       'linear': tf.keras.activations.linear,
                       'softmax': tf.keras.activations.softmax,
                       'none': None}
       return activation_dict[name]

class CWGAN(object):
        def __init__(self, embedding_dim = 30, condition = True, hidden_layer_sizes=(100,100), layer_norm = False, activation_name = 'relu', dropout = 0
                    , n_cross_layers = 1, hard_sampling = True, cat_activation_name = 'softmax', num_activation_name = 'none', condition_num_on_cat = True,
                        reduce_cat_dim = True, use_num_hidden_layer = True, 
                        d_embedding_dims_sizes = 'auto', noisy_num_cols= True, d_n_cross_layers = 1, d_hidden_layer_sizes = (100, 100), d_activation_name = 'relu',
                        d_sigmoid_activation = True, d_layer_norm = True,
                        A_embedding_dims_sizes = 'auto', A_hidden_layer_sizes = (64,64), A_n_cross_layers = 1, A_sigmoid_activation = True):
            self._embedding_dim = embedding_dim     #noise_dim
            self._condition = condition
            self._hidden_layer_sizes = hidden_layer_sizes
            self._layer_norm = layer_norm
            self._activation = get_activation_by_name(activation_name)
            self._dropout = dropout
            self.n_cross_layers = n_cross_layers
            self.hard_sampling = hard_sampling
            self.cat_activation = get_activation_by_name(cat_activation_name)
            self.num_activation = get_activation_by_name(num_activation_name)
            self.training = True              #?????????????????????
            self.condition_num_on_cat = condition_num_on_cat
            self.reduce_cat_dim = reduce_cat_dim
            self.use_num_hidden_layer = use_num_hidden_layer

            self.d_embedding_dims_sizes = d_embedding_dims_sizes
            self.noisy_num_cols = noisy_num_cols
            self.d_n_cross_layers = d_n_cross_layers
            self.d_hidden_layer_sizes = d_hidden_layer_sizes
            self.d_activation = get_activation_by_name(d_activation_name)
            self.sigmoid_activation = d_sigmoid_activation
            self.d_layer_norm = d_layer_norm

            self.A_embedding_dims_sizes = A_embedding_dims_sizes
            self.A_hidden_layer_sizes = A_hidden_layer_sizes
            self.A_n_cross_layers = A_n_cross_layers
            self.A_sigmoid_activation = A_sigmoid_activation
            


        def make_generator(self, transformer):
            input_to_hidden_layers_dim = self._embedding_dim + 1 if self._condition else self._embedding_dim
            dim_to_final_layer = input_to_hidden_layers_dim
            #input to generator
            inp_gen = tf.keras.layers.Input(shape=(input_to_hidden_layers_dim,))

            #hidden layers
            if len(self._hidden_layer_sizes) > 0:
                input_dim = input_to_hidden_layers_dim
                output_dim = self._hidden_layer_sizes[0]
                x = Dense(input_dim, output_dim)(inp_gen)
                if self._layer_norm:
                    for i in range(len(self._hidden_layer_sizes) - 1):
                        x = Dense(self._hidden_layer_sizes[i], self._hidden_layer_sizes[i+1])(x)
                        x = tf.keras.layers.LayerNormalization()(x)
                else: 
                    for i in range(len(self._hidden_layer_sizes) - 1):
                        x = Dense(self._hidden_layer_sizes[i], self._hidden_layer_sizes[i+1])(x)
                dim_to_final_layer = self._hidden_layer_sizes[-1] 

            #apply activation function and dropout to the output of hidden layers 
            if self._activation is not None:
                x = self._activation(x)
            if self._dropout > 0:
                x = tf.keras.layers.Dropout(rate=self._dropout)(x, training=True)
            
            #cross layers
            if self.n_cross_layers > 0:
                input_to_cross_layers_dim = input_to_hidden_layers_dim
                x0 = inp_gen
                x_cross = inp_gen

            for i in range(self.n_cross_layers):
                x_cross = Cross(input_to_cross_layers_dim)(x0, x_cross)

            #concat the output of cross layers and the output of hidden layers
            x = tf.concat([x, x_cross], axis = 1)     #outputs 
            dim_to_final_layer += input_to_cross_layers_dim

            #get categorical output
            cat_output_dims = transformer.cat_dims
            x_cat = []
            for dim in cat_output_dims:
                cat = Dense(dim_to_final_layer ,dim)(x)
                if self.cat_activation is not None:
                    cat = self.cat_activation(cat)
                    #elif self.cat_activation == 'gumbel_softmax':
                    #cat = tf.keras.activations.softmax(cat)            #???????????????????????????
                x_cat.append(cat)
            if self.hard_sampling and not self.training:          #???????????? self.training
                # turn probs into onehot (draws from the distribution)
                x_cat = [tf.one_hot(tf.random.categorical(logits, 1), depth=logits.shape[1]) for logits in x_cat]
            
            # condition numerical on categorical
            if self.condition_num_on_cat:
                x_cat_cond = [tf.identity(_tensor) for _tensor in x_cat]
                if self.reduce_cat_dim:
                    # embed each onehot vector individually
                    x_emb = []
                    x_emb_concatenated_dim = 0
                    for idx, cat_dim in enumerate(cat_output_dims):
                        emb_dim = int(min(np.ceil(cat_dim / 3), 20))
                        emb = Dense(cat_dim , emb_dim)(x_cat[idx])
                        x_emb.append(emb)
                        x_emb_concatenated_dim += emb_dim
                    x_emb_concatenated = tf.concat(x_emb, axis=1)
                    emb  = Dense(x_emb_concatenated_dim, 16)(x_emb_concatenated)
                    if self._activation is not None:
                        emb = self._activation(emb)
                    x_cat_cond = [emb]
                if not self.reduce_cat_dim:
                    dim_to_final_layer += sum(cat_output_dims)
                else:
                    dim_to_final_layer += 16
                
                x  = tf.concat([x, *x_cat_cond], axis=1)

            if self.use_num_hidden_layer:
                x = Dense(dim_to_final_layer, 32)(x)
                if self._activation is not None:
                   x = self._activation(x)
                dim_to_final_layer = 32

            output_dim = transformer.n_numCols
            x_num = Dense(dim_to_final_layer, output_dim)(x)
            if self.num_activation is not None:
                x_num = self.num_activation(x_num)

            x_out = tf.concat([x_num, *x_cat], axis=1)
            
            generator = tf.keras.models.Model(inp_gen, x_out)

            return generator


        def make_discriminator(self, transformer):
            cat_input_dims = transformer.cat_dims
            num_input_dim = transformer.n_numCols
            #dimension of input to discriminator
            inp_disc_dim = num_input_dim + sum(cat_input_dims)
            #input to discriminator
            inp_disc = tf.keras.layers.Input(shape=(inp_disc_dim,))
            #split input to discriminator
            x_num, x_cat = tf.split(inp_disc, num_or_size_splits=[num_input_dim, sum(cat_input_dims)], axis=1)
            if self.noisy_num_cols:
                # Create a normal distribution with mean 0 and standard deviation 0.01
                dist = tf.random.normal(shape=tf.shape(x_num), mean=0.0, stddev=0.01)
                # Add the normal distribution to the original tensor
                x_num = x_num + dist
            if self.d_embedding_dims_sizes is not None and cat_input_dims is not None:
                if self.d_embedding_dims_sizes == 'auto':
                    self.d_embedding_dims_sizes = [int(min(np.ceil(cat_dim/3), 20)) for cat_dim in cat_input_dims]
                dim_to_final_cat = sum(self.d_embedding_dims_sizes)
                start_idx = 0
                x_emb = []
                for emb_dim, cat_dim in zip(self.d_embedding_dims_sizes, cat_input_dims):
                    end_idx = start_idx + cat_dim
                    partition = x_cat[:, start_idx:end_idx]
                    emb = Dense(cat_dim, emb_dim)(partition)
                    x_emb.append(emb)
                    start_idx = end_idx
                x = tf.concat([x_num, *x_emb], axis=1)
            elif cat_input_dims is not None:
                dim_to_final_cat = sum(cat_input_dims) if len(cat_input_dims) > 0 else 0
                x = tf.concat([x_num, x_cat], axis=1)
            else:
                dim_to_final_cat = 0
                x = x_num

            input_to_final_dim = num_input_dim + 1 if self._condition else num_input_dim
            input_to_final_dim += dim_to_final_cat
            input_to_cross_layers = input_to_final_dim
            if self._condition:
                # append y to X if conditioning
                input_condition = tf.keras.layers.Input((1,))
                #y_t = sampler.sample_condition(x.shape[0])
                x = tf.concat([x, input_condition], axis=1)
            if self.d_n_cross_layers > 0:
                x0 = x
            if len(self.d_hidden_layer_sizes) > 0:
                x = Dense(input_to_final_dim, self.d_hidden_layer_sizes[0])(x)
                if self.d_layer_norm:
                   for i in range(len(self.d_hidden_layer_sizes) - 1):
                       x = Dense(self.d_hidden_layer_sizes[i], self.d_hidden_layer_sizes[i+1])(x)
                       x = tf.keras.layers.LayerNormalization()(x)
                else:
                    for i in range(len(self.d_hidden_layer_sizes) - 1):
                       x = Dense(self.d_hidden_layer_sizes[i], self.d_hidden_layer_sizes[i+1])(x)
                input_to_final_dim = self.d_hidden_layer_sizes[-1]
            if self.d_activation is not None:
                x = self.d_activation(x)
            if self.d_n_cross_layers > 0:
                x_cross = x0
                for i in range(self.d_n_cross_layers):
                    x_cross = Cross(input_to_cross_layers)(x0, x_cross)
                input_to_final_dim += input_to_cross_layers
                x = tf.concat([x, x_cross], axis = 1)
            x = Dense(input_to_final_dim, 1)(x)
            if self.sigmoid_activation:
                x = tf.sigmoid(x)
            if self._condition:
               discriminator = tf.keras.models.Model(inputs = [inp_disc,input_condition], outputs = x)
            else:
               discriminator = tf.keras.models.Model(inputs = inp_disc , outputs = x)
            return discriminator
       
        def make_Aux_classifier(self, transformer):
            if not self._condition:
                cat_input_dims = transformer.cat_dims[:-1]
            else:
                cat_input_dims = transformer.cat_dims
            num_input_dim = transformer.n_numCols
            #dimension of input to discriminator
            inp_disc_dim = num_input_dim + sum(cat_input_dims)
            #input to discriminator
            inp_disc = tf.keras.layers.Input(shape=(inp_disc_dim,))
            #split input to discriminator
            x_num, x_cat = tf.split(inp_disc, num_or_size_splits=[num_input_dim, sum(cat_input_dims)], axis=1)
            if self.noisy_num_cols:
                # Create a normal distribution with mean 0 and standard deviation 0.01
                dist = tf.random.normal(shape=tf.shape(x_num), mean=0.0, stddev=0.01)
                # Add the normal distribution to the original tensor
                x_num = x_num + dist
            if self.A_embedding_dims_sizes is not None and cat_input_dims is not None:
                if self.A_embedding_dims_sizes == 'auto':
                    self.A_embedding_dims_sizes = [int(min(np.ceil(cat_dim/3), 20)) for cat_dim in cat_input_dims]
                dim_to_final_cat = sum(self.A_embedding_dims_sizes)
                start_idx = 0
                x_emb = []
                for emb_dim, cat_dim in zip(self.A_embedding_dims_sizes, cat_input_dims):
                    end_idx = start_idx + cat_dim
                    partition = x_cat[:, start_idx:end_idx]
                    emb = Dense(cat_dim, emb_dim)(partition)
                    x_emb.append(emb)
                    start_idx = end_idx
                x = tf.concat([x_num, *x_emb], axis=1)
            elif cat_input_dims is not None:
                dim_to_final_cat = sum(cat_input_dims) if len(cat_input_dims) > 0 else 0
                x = tf.concat([x_num, x_cat], axis=1)
            else:
                dim_to_final_cat = 0
                x = x_num

            input_to_final_dim = num_input_dim 
            input_to_final_dim += dim_to_final_cat
            input_to_cross_layers = input_to_final_dim
            if self.A_n_cross_layers > 0:
                x0 = x
            if len(self.A_hidden_layer_sizes) > 0:
                x = Dense(input_to_final_dim, self.A_hidden_layer_sizes[0])(x)
                if self.d_layer_norm:
                   for i in range(len(self.A_hidden_layer_sizes) - 1):
                       x = Dense(self.A_hidden_layer_sizes[i], self.A_hidden_layer_sizes[i+1])(x)
                       x = tf.keras.layers.LayerNormalization()(x)
                else:
                    for i in range(len(self.A_hidden_layer_sizes) - 1):
                       x = Dense(self.A_hidden_layer_sizes[i], self.A_hidden_layer_sizes[i+1])(x)
                input_to_final_dim = self.A_hidden_layer_sizes[-1]
            if self.d_activation is not None:
                x = self.d_activation(x)
            if self.A_n_cross_layers > 0:
                x_cross = x0
                for i in range(self.A_n_cross_layers):
                    x_cross = Cross(input_to_cross_layers)(x0, x_cross)
                input_to_final_dim += input_to_cross_layers
                x = tf.concat([x, x_cross], axis = 1)
            x = Dense(input_to_final_dim, 1)(x)
            if self.A_sigmoid_activation:
                x = tf.sigmoid(x)
            discriminator = tf.keras.models.Model(inputs = inp_disc, outputs = x)
            return discriminator
