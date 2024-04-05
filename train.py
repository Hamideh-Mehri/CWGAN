import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

class Train(object):
    def __init__(self, transformer, cwgan, batch_size,epoch, generator_lr=1e-4, generator_decay=1e-6, discriminator_lr=2e-4, auxclassifier_lr = 1e-4,
                 discriminator_decay=1e-6, discriminator_steps=1, normal_noise = True, aux_batchsize = 100, aux_epochs = 30):
        self._embedding_dim = cwgan._embedding_dim
        self._batch_size = batch_size
        self._normal_noise = normal_noise
        self._condition = cwgan._condition
        self.generator = cwgan.make_generator(transformer)
        self. discriminator = cwgan.make_discriminator(transformer)
        self.aux_classifier = cwgan.make_Aux_classifier(transformer)
        self.transformer = transformer
        self._epochs = epoch
        self._generator_lr = generator_lr
        self._discriminator_lr = discriminator_lr
        self._auxclassifier_lr = auxclassifier_lr
        self._discriminator_steps = discriminator_steps
        self._generator_decay = generator_decay
        self._discriminator_decay = discriminator_decay
        self.aux_batchsize = aux_batchsize
        self.aux_epochs = aux_epochs

    def sample_condition(self, y):
        # sample y for generator
         if isinstance(y, (np.ndarray, tf.Tensor)):
            y = tf.convert_to_tensor(y)
         elif y == '50-50':
            n_ones = self._batch_size // 2
            n_zeros = self._batch_size - n_ones
            y = tf.concat([tf.zeros((n_zeros, 1)), tf.ones((n_ones, 1))], axis=0)
         elif isinstance(y, (int, float)):
            if y == 1:
               y = tf.ones((self._batch_size, 1))
            elif y == 0:
               y = tf.zeros((self._batch_size, 1))
         # y \in {0,1} > {-1,1}
         y_t = (y + (y - 1))

         return y_t 
    def pretrain_aux(self,X_tens, y_tens):
        optimizerA = tf.keras.optimizers.Adam(learning_rate = self._auxclassifier_lr, beta_1=.0, beta_2=0.9)
        
        iters_per_epoch = int(np.ceil(X_tens.shape[0] / self.aux_batchsize)) 
        if not self._condition:
           y = tf.reshape(tf.gather(tf.transpose(X_tens), tf.shape(X_tens)[1]-2), (-1, 1))
           X = X_tens[:, :-2]
        else:
           y = y_tens
           X = X_tens
        for epoch in range(self.aux_epochs):
            permutation = tf.random.shuffle(tf.range(tf.shape(X)[0]))
            shuffled_x = tf.gather(X, permutation)
            shuffled_y = tf.gather(y, permutation)
            for batch_idx in range(iters_per_epoch):
                X_batch = shuffled_x[batch_idx * self.aux_batchsize:(batch_idx + 1) * self.aux_batchsize]
                y_batch = shuffled_y[batch_idx * self.aux_batchsize:(batch_idx + 1) * self.aux_batchsize]
                with tf.GradientTape() as tape:
                   output = self.aux_classifier(X_batch, training = True)
                   loss = tf.keras.losses.binary_crossentropy(output, y_batch)
                   Aux_loss = tf.reduce_mean(loss)
                grads_aux = tape.gradient(Aux_loss, self.aux_classifier.trainable_variables)
                optimizerA.apply_gradients(zip(grads_aux, self.aux_classifier.trainable_variables))
            print('epoch {}'.format(epoch))
     
        preds = self.aux_classifier(X).numpy()
        print(f'ACC: {accuracy_score(y[:, -1], np.where(preds > 0.5, 1, 0)):.4f} '
                     f'AUC: {roc_auc_score(y[:, -1], preds):.4f} '
                     f'BCE: {log_loss(y[:, -1], preds):.4f}')

    def compute_aux_classifier_loss(self, fake, y_batch):
         if not self._condition:
            fake_y = tf.reshape(tf.gather(tf.transpose(fake), tf.shape(fake)[1]-1), (-1, 1))
            fake_X = fake[:, :-2]
         else:
            fake_y = y_batch
            fake_X = fake
         aux_output = self.aux_classifier(fake_X)
         aux_loss = tf.keras.losses.binary_crossentropy(aux_output, fake_y)
         aux_loss = tf.clip_by_value(aux_loss, clip_value_min=0.3, clip_value_max=tf.reduce_max(aux_loss))
         aux_loss = tf.reduce_mean(aux_loss)
         return aux_loss
        

    def synthesise_data(self, n, condvec):
        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = tf.zeros(shape=(self._batch_size, self._embedding_dim), dtype=tf.float32)
            std = mean + 1
            if self._normal_noise:
               #random numbers drawn from a normal distribution
               fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
            else: 
               #random numbers drawn from a uniform distribution
               fakez = tf.random.uniform(shape=(self._batch_size, self._embedding_dim)) 
            if self._condition:
               y = self.sample_condition(condvec)
               fakez = tf.concat([fakez,y], axis=1)
            fake = self.generator(fakez)
            data.append(fake.numpy())
        data = np.concatenate(data, axis=0)
        return data

    def train(self):
        
        mean = tf.zeros(shape=(self._batch_size, self._embedding_dim), dtype=tf.float32)
        std = mean + 1
        X_tens, y_tens, _ = self.transformer.transformData()
        steps_per_epoch = max(tf.shape(X_tens)[0]// self._batch_size, 1)
        optimizerG = tf.keras.optimizers.Adam(learning_rate = self._generator_lr, beta_1=.0, beta_2=0.9, decay = self._generator_decay)
        optimizerD = tf.keras.optimizers.Adam(learning_rate = self._discriminator_lr, beta_1=.0, beta_2=0.9, decay = self._discriminator_decay)
        
        #pretrain aux-classifier
        self.pretrain_aux(X_tens, y_tens)
        
        for i in range(self._epochs):
            permutation = tf.random.shuffle(tf.range(tf.shape(X_tens)[0]))
            shuffled_x = tf.gather(X_tens, permutation)
            if y_tens is not None:
               shuffled_y = tf.gather(y_tens, permutation)
            else:
               shuffled_y = None
            for id_ in range(steps_per_epoch - 1):
                #sample noise
                if self._normal_noise:
                   #random numbers drawn from a normal distribution
                   fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
                else: 
                   #random numbers drawn from a uniform distribution
                   fakez = tf.random.uniform(shape=(self._batch_size, self._embedding_dim)) 
                X_batch = self.transformer.getbatchX(shuffled_x, self._batch_size, id_)
                y_batch = self.transformer.getbatchY(shuffled_y, self._batch_size, id_)
                if self._condition:
                    y = self.sample_condition(y_batch)   
                    fakez = tf.concat([fakez, y], axis=1)
                else:
                    y = None
                #generate batch of data
                fake = self.generator(fakez)
                #train discriminator
                with tf.GradientTape() as tape:
                    if self._condition:
                        y_fake = self.discriminator([fake, y], training=True)
                        y_real = self.discriminator([[X_batch, y]], training=True)
                    else:
                        y_fake = self.discriminator(fake, training=True)
                        y_real = self.discriminator(X_batch, training=True)
                    y_real = tf.reshape(y_real, [-1])
                    y_fake = tf.reshape(y_fake, [-1])
                    loss_D_real = tf.keras.losses.binary_crossentropy(tf.ones_like(y_real), y_real)
                    loss_D_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(y_fake), y_fake)
                    disc_loss = loss_D_real + loss_D_fake
                grads_disc = tape.gradient(disc_loss, self.discriminator.trainable_variables)
                optimizerD.apply_gradients(zip(grads_disc, self.discriminator.trainable_variables))
                if id_ % self._discriminator_steps == 0:
                    #train generator
                    if self._normal_noise:
                       #random numbers drawn from a normal distribution
                       fakez = tf.random.normal(shape=(self._batch_size, self._embedding_dim), mean=mean, stddev=std)
                    else: 
                       #random numbers drawn from a uniform distribution
                       fakez = tf.random.uniform(shape=(self._batch_size, self._embedding_dim)) 
                    if self._condition:
                       fakez = tf.concat([fakez, y], axis=1)
                    #generate batch of data
                    #fake = self.generator(fakez)
                    with tf.GradientTape() as tape:
                       fake = self.generator(fakez)
                       if self._condition:
                          y_fake = self.discriminator([fake, y], training=True)
                       else:
                          y_fake = self.discriminator(fake, training=True)
                       aux_clf_loss = self.compute_aux_classifier_loss(fake, y_batch)
                       gen_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(y_fake), y_fake) + 0.1 * aux_clf_loss
                    grads_gen = tape.gradient(gen_loss, self.generator.trainable_variables)
                    optimizerG.apply_gradients(zip(grads_gen, self.generator.trainable_variables))