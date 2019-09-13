#%%
import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

#%%
# Get an example dataset

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0

#%%
# Define a simple sequential model

def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


#%%
# Save checkpoints during training

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves teh model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)

# Train the model with the new callback
model.fit(train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
        callbacks=[cp_callback])    # Pass callback to training


#%%
!ls {checkpoint_dir}


#%%
# Rebuild a fresh, untrained model

# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

#%%
# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

#%%
# Checkpoint callback options

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5
)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images,
            train_labels,
            epochs=50,
            callbacks=[cp_callback],
            validation_data=(test_images, test_labels),
            verbose=0)


#%%
! ls {checkpoint_dir}

#%%
latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# Note: the default tensorflow format only saves the 5 most recent checkpoints

#%%
# To test, reset the model and load the latest checkpoint

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


#%%
# Manually save weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


#%%
# Save the entire model
# - The weight values
# - The model's configuration(architecture)
# - The optimizer configuration

# you can load them in Tensorflow.js (HDF5) and then train an run them in web browsers, or convert them to run on mobile devices using Tensorflow Lite

# Create a new model instance
model = create_model()

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Save teh entire model to a HDF5 file
model.save('my_model.h5')


#%%
# Recreate the exact same model, including the weights and the optimizer
new_model = keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()


#%%
# Check its accuracy

loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


#%%
