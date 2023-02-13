import numpy as np
def sigmoid(x):
 return (1/(1+np.exp(-x)))
trainX=np.array([[0,0,1],
 [1,1,1],
 [1,0,1],
 [0,1,1]])
trainY=np.array([[0,1,1,0]]).T
weights=np.array([0.15,0.20,0.25])
print(weights)
for i in range(1):
 input_layer=trainX
 output=sigmoid(np.dot(input_layer,weights))
print("Outputs after training")
print(output)
# Calculate the total loss
error = trainY - output
squared_error = np.square(error)
mean_squared_error = np.mean(squared_error)
total_loss = mean_squared_error
print("Total Loss: ", total_loss)