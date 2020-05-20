import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SCC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epochs = 25


data_1 = {
    'train':'./linear_data/train.csv',
    'test':'./linear_data/test.csv'
}

data_2 = {
    'train':'./quadratic_data/train.csv',
    'test':'./quadratic_data/test.csv'
}

data_3 = {
    'train':'./complex_data/train.csv',
    'test':'./complex_data/test.csv'
}

data = data_3

train_df = pd.read_csv(data['train'])
test_df = pd.read_csv(data['test'])
inputs_test = np.column_stack((test_df['x'].values,test_df['y'].values))

model= Sequential([
                Dense(512,input_shape=(2,),activation='relu'),
                Dense(256,activation='relu'),
                LeakyReLU(alpha='0.01'),
                Dense(128,activation='relu'),
                Dense(64,activation='relu'),
                Dropout(0.1),
                
                
                Dense(2,activation='sigmoid')
                ])


model.compile(optimizer='adam',
            loss=SCC(from_logits=True),
            metrics=['accuracy'])

inputs=np.column_stack((train_df['x'].values,train_df['y'].values))



before_training_predictions = model.predict(inputs_test)


evolution = model.fit(inputs,train_df['color'].values, batch_size=32,epochs=epochs)



predictions = model.predict(inputs_test)


def check_predictions(predictions):
    predicted_outputs = [0 if p[0] > p[1] else 1 for p in predictions]
    verified_outputs = predicted_outputs == test_df['color'].values

    markers = ['o' if vo == True else 'x' for vo in verified_outputs]
    colors = ['r' if po == 0 else 'b' for po in predicted_outputs]

    outputs = np.column_stack((inputs_test,markers,colors))

    corect = outputs[outputs[:,2]=='o']
    wrong = outputs[outputs[:,2]=='x']

    return (corect,wrong)




fig,((expected_g,before_g),(actiune_g,evolution_g)) = plt.subplots(2,2)
#evolution

x = np.arange(start = 0,stop = epochs,step=1)
y = evolution.history['loss']
evolution_g.plot(x,y)

y = evolution.history['acc']
evolution_g.plot(x,y)
evolution_g.set_title('Evolution and loss')
#before
corect,wrong = check_predictions(before_training_predictions)
x = np.float32(corect[:,0])
y = np.float32(corect[:,1])
colors =corect[:,3]
before_g.scatter(x,y,marker='o',color=colors)


x = np.float32(wrong[:,0])
y = np.float32(wrong[:,1])
colors =wrong[:,3]
before_g.scatter(x,y,marker='x',color='k')
before_g.set_title('Before Training')

#action
corect,wrong = check_predictions(predictions)

x = np.float32(corect[:,0])
y = np.float32(corect[:,1])
colors =corect[:,3]
actiune_g.scatter(x,y,marker='o',color=colors)

x = np.float32(wrong[:,0])
y = np.float32(wrong[:,1])
colors =wrong[:,3]
actiune_g.scatter(x,y,marker='x',color='k')
actiune_g.set_title('After Training')

#expected
colors = ['r' if t == 0 else 'b' for t in test_df['color']]
expected_g.scatter(test_df['x'],test_df['y'],marker='o',color=colors)
expected_g.set_title('Expected')


plt.show()

