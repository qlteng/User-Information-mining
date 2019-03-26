import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, GRU, multiply, concatenate, Activation, Masking, Reshape, add
from keras.layers import Bidirectional, Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, MaxPooling1D
from keras.utils.vis_utils import plot_model
from utils.layer_utils import AttentionLSTM


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input._keras_shape[-1] # channel_axis = -1 for TF
    # se = Conv1D(1, 1, padding='same', activation='relu', kernel_initializer='he_uniform')(input)
    se = GlobalAveragePooling1D()(input)
    # print (se)
    se = Activation('softmax')(se)
    # se = Reshape([-1])(se)
    # # print (se)
    # se = Dense(12,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)

    # se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se

n_channel = 6
n_steps = 128
n_class = 6

# ip = Input(shape=(n_channel, n_steps))
# y = AttentionLSTM(32)(ip)

ip = Input(shape=(n_steps, n_channel))
# y=AttentionLSTM(32)(ip)
y = Conv1D(2 * n_channel, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(ip)
y = squeeze_excite_block(y)

# y = Conv1D(2 * n_channel, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
# y = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(y)
# y = Conv1D(4 * n_channel, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
# y = Conv1D(4 * n_channel, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
# y = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(y)
# # y = Permute((2,1))(y)
# y = LSTM(32, return_sequences= True)(y)
# # y = Dropout(0.5)(y)
# y = LSTM(32, return_sequences= False)(y)
# # y = squeeze_excite_block(y)
# y = Conv1D(3 * n_channel, 3, padding='same', activation='relu', kernel_initializer='he_uniform')(y)
# y = MaxPooling1D(pool_size = 2, strides = 2, padding = 'same')(ip)
# y = squeeze_excite_block(y)

#
# y = Permute((2, 1))(y)
# y = GRU(32, return_sequences= True)(ip)
# y = Dropout(0.5)(y)
# y = GRU(32, return_sequences= False)(y)
# y = Reshape([-1])(y)
y = Dense(32, activation='sigmoid')(y)
# y = Dropout(0.5)(y)
out = Dense(n_class, activation='softmax')(y)
model = Model(ip, out)
plot_model(model, to_file='./model_structure/simple-cnnattention.jpg', show_shapes=True)