import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.w1 = self.w2 = 1
        self.bias = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def log_loss(self, y_predicted):
        epsilon = 1e-15
        y_pred = [max(i,epsilon) for i in y_predicted]
        y_pred = [min(i,1-epsilon) for i in y_pred]
        y_pred = np.array(y_pred)
        return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    
    def predict(self, age, affordibility):
        weighted_sum  = self.w1 * age + self.w2 * affordability + self.bias
        return self.sigmoid(weighted_sum)
    
    def fit(self, age, affordability, y_true, epochs):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = self.sigmoid(weighted_sum)
            
            loss = self.log_loss(y_predicted)
            
            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
            w2d = (1/n)*np.dot(np.transpose(affordability),(y_predicted-y_true))
            
            bias_d = np.mean(y_predicted-y_true)
            
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            print(f"Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}")
        return self.w1, self.w2, self.bias
    
age = np.arange(20, 50)
affordability = np.array([1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1])
y_true = np.array([1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1])

network = NeuralNetwork()
network.fit(age, affordability, y_true, 500)

a = np.arange(15, 45)
afford = np.array([0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1])
pred = network.predict(a, afford)
print(pred)