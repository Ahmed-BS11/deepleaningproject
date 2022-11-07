import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow import keras


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

#load data
iris=load_iris()
X = iris.data
y = iris.target
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(300, input_shape=(4,), activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(500, activation="relu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(3, activation='softmax')
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss= keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

history=model.fit(X_train,y_train,batch_size=5,epochs=100)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

results = model.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

class_names=iris.target_names
X_new=X_test[:5]

y_preddd=model.predict(X_new)
y_pred=np.argmax(y_preddd,axis=-1)
print(np.array(class_names)[y_pred])

y_new = y_test[:5]
print(np.array(class_names)[y_new])

print("ahmed")



