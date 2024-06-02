import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import reuters
from keras.callbacks import EarlyStopping
from keras.engine.topology import Layer

class self_attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(self_attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(self_attention, self).build(input_shape)
    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        V = K.batch_dot(QK, WV)
        return V
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim
        })
        return config

class customvariationallayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
    def vae_loss(self, x, inference_x):
        decoder_loss = K.sum(x * K.log(inference_x), axis=-1)
        encoder_loss = -0.5 * (K.sum(inv_sigma1 * K.exp(z_log_var) + K.square(z_mean) * inv_sigma1 - 1 - z_log_var, axis=-1) + log_det_sigma)
        return -K.mean(encoder_loss + decoder_loss)
    def call(self, inputs):
        x = inputs[0]
        inference_x = inputs[1]
        loss = self.vae_loss(x, inference_x)
        self.add_loss(loss, inputs=inputs)
        return x

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batchsize, num_topic), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

def ntm_loss(x, inference_x):
    decoder_loss = K.sum(x * K.log(inference_x), axis=-1)
    encoder_loss = -0.5 * (K.sum(inv_sigma1 * K.exp(z_log_var) + K.square(z_mean) * inv_sigma1 - 1 - z_log_var, axis=-1) + log_det_sigma)
    return -K.mean(encoder_loss + decoder_loss)

def attention(z, u, t, alpha):
    z_tanh = Dense(units=16)(z)
    z_tanh = Activation('tanh')(z_tanh)
    u = tf.expand_dims(u, axis=1)
    weights = tf.matmul(u, z_tanh)
    weights = Softmax()(weights)
    weights = tf.multiply(weights, tf.expand_dims(t - alpha, axis=1))
    output = tf.matmul(weights, z_tanh, transpose_b=True)
    return output

x_train, t_train, y_train, x_test, t_test, y_test = load_data(i_fold, n_fold)
V, num_hidden, num_topic, alpha = 116, 100, 16, 1. / 20
mu1 = np.log(alpha) - 1 / num_topic * num_topic * np.log(alpha)
sigma1 = 1. / alpha * (1 - 2. / num_topic) + 1 / (num_topic ** 2) * num_topic / alpha
log_det_sigma = num_topic * np.log(sigma1)
inv_sigma1 = 1. / sigma1

inputs_ntm = Input(shape=(V,))
h = Dense(num_hidden, activation='softplus')(inputs_ntm)
h = Dense(num_hidden, activation='softplus')(h)
z_mean = BatchNormalization()(Dense(num_topic)(h))
z_log_var = BatchNormalization()(Dense(num_topic)(h))
unnormalized_z = Lambda(sampling, output_shape=(num_topic,))([z_mean, z_log_var])
theta = Activation('softmax')(unnormalized_z)
theta = Dropout(0.5)(theta)
v = Dense(units=num)(theta)
v = BatchNormalization()(v)
v = Activation('softmax')(v)
doc = Dense(units=V)(v)
doc = BatchNormalization()(doc)
output_ntm = Activation('softmax', name='ntm_output')(doc)
output = customVariationalLayer()([inputs_ntm, doc])

inputs_attention = Input(shape=(14, 9))
x1 = self_attention(num_units)(inputs_attention)
x2 = keras.layers.Dense(num_time, activation='relu')(v)
d = K.constant(0.1, dtype=tf.float32, shape=(1, num_units))
x = attention(x1, x2, unnormalized_z, d)
x = Flatten()(x)
x = keras.layers.Dense(16, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(8, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
output_attention = keras.layers.Dense(2, activation='softmax', name='attention_output')(x)

model = Model(inputs=[inputs_attention, inputs_ntm], outputs=[output_attention, output_ntm])
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.99),
    loss={'attention_output': 'categorical_crossentropy', 'ntm_output': ntm_loss},
    loss_weights={'attention_output': 1, 'ntm_output': 0.1},
    metrics={'attention_output': 'accuracy'})
history = model.fit(x=[x_train, t_train], y=[y_train, t_train], validation_data=([x_test, t_test], [y_test, t_test]), epochs=50, batch_size=64)
y_pred, t_pred = model.predict([x_test, topic_test])
acc, pre, rec, f1 = evaluation_indicator(y_true=y_test, y_pred=y_pred)
