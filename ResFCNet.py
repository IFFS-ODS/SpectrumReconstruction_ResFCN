import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import datetime




def identity_block(input_tensor,units):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		units:output shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(input_tensor)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(x)

	x = layers.add([x, input_tensor])
	x = layers.Activation('relu')(x)

	return x

def dens_block(input_tensor,units):
	"""A block that has a dense layer at shortcut.
	# Arguments
		input_tensor: input tensor
		unit: output tensor shape
	# Returns
		Output tensor for the block.
	"""
	x = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(input_tensor)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(x)
	x = layers.Activation('relu')(x)

	x = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(x)

	shortcut = layers.Dense(units, kernel_regularizer= regularizers.L2(0.001))(input_tensor)

	x = layers.add([x, shortcut])
	x = layers.Activation('relu')(x)
	return x


def ImprovedResNet():
    with tf.device('/GPU:0'):
        Res_input = layers.Input(shape=(16,))
        x = layers.Dense(64, kernel_regularizer= regularizers.L2(0.001))(Res_input)
        x = layers.Activation('relu')(x)
        
        x = dens_block(x,256)
        x = identity_block(x,256)
        x = identity_block(x,256)
        
        x = dens_block(x,512)
        x = identity_block(x,512)
        x = identity_block(x,512)

        x = dens_block(x,1024)
        x = identity_block(x,1024)
        x = identity_block(x,1024)

        x = layers.Dense(1024, activation='linear')(x)
        x = layers.Dense(2048, activation='linear')(x)
        x = layers.Dense(1650, activation='linear')(x)
        model = models.Model(inputs=Res_input, outputs=x)

    return model
#Loss 
def custom_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    l1_loss = tf.keras.backend.sum(tf.keras.backend.abs(y_true - y_pred))
    return mse_loss + 1e-3*l1_loss
# learning rate
def lr_schedule(epoch):
    if epoch < 20:
        return 1e-3
    elif 20 <= epoch < 40:
        return 1e-4
    elif 40 <= epoch < 60:
        return 1e-5
    else:
        return 1e-6

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#################################Prepare data####################################
plt.switch_backend('agg')
path_feature = 'feature_16.txt'
path_label = 'label_1650.txt'

featureSet = np.loadtxt(path_feature)
featureSet = np.array(featureSet)
labelSet = np.loadtxt(path_label)
labelSet = labelSet
print(featureSet.shape)
print(labelSet.shape)
x = featureSet
y = labelSet

xscale = x
yscale = y

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale,test_size=0.20)

X_train = tf.constant(X_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)
X_test = tf.constant(X_test, dtype=tf.float32)
y_test = tf.constant(y_test, dtype=tf.float32)

##############################Build Model###############################
model = ImprovedResNet()
optimizer = Adam(learning_rate=1e-4)
model.compile(loss=custom_loss, optimizer=optimizer, metrics=['mse'])
model.summary()

#compute running time
starttime = datetime.datetime.now()

lr_callback = LearningRateScheduler(lr_schedule)

history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2, callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=30,verbose=2, mode='auto'), lr_callback], validation_split=0.1)
endtime = datetime.datetime.now()

##############################Save Model#################################
model.save('OptimalModel.h5')

#############################Model Predicting############################
yhat = model.predict(X_test)

print('The time cost: ')
print(endtime - starttime)
print('The test loss: ')
print(mean_squared_error(yhat,y_test))
print(r2_score(yhat,y_test))


###############################Visualize Model############################
# "Loss"
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('loss.png')

# Find the indices of the five lowest validation losses
lowest_val_loss_indices = sorted(enumerate(history.history['val_loss']), key=lambda x: x[1])[:12]
lowest_val_loss_indices = [index for index, _ in lowest_val_loss_indices]

# Plot the graphs for the five sets with the lowest validation losses
for i in lowest_val_loss_indices:
    plt.figure()
    plt.plot(y_test[i])
    plt.plot(yhat[i])
    plt.title(f'Result for ResNet Regression (Index {i})')
    plt.ylabel('Y value')
    plt.xlabel('Instance')
    plt.legend(['Real value', 'Predicted Value'], loc='upper right')
    plt.savefig(f'Output_figure_{i}.png')
