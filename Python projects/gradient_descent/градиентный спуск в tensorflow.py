import tensorflow as tf
import matplotlib.pyplot as plt

# Generate some random data
tf.random.set_seed(23)
x = tf.random.uniform(
    shape = (100,1),
    minval=0,
    maxval=100,
    dtype=tf.dtypes.float32,
)
  
y =2*x + tf.random.normal(shape = (100,1),
                     mean=50.0, 
                     stddev=20, 
                     dtype=tf.dtypes.float32
                             )
  
plt.scatter(x,y)
plt.show()


# Define the weight and bias for model
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")
print('Weight :',W)
print('Bias   :',b)

# Define linear
def linear_regression(x):
    return W * x + b

# Define the cost function
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define the optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.00001)

# Define the training loop
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = linear_regression(x)
        loss = mean_squared_error(y, y_pred)
    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))
    return loss

# Train the model
# plt.figure(figsize=(15,7))
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), dpi=200)
fig2, (ax) = plt.subplots(1, figsize=(7, 5))
for i in range(50):
    loss = train_step(x, y)
    ax1.plot(i, W, 'b*')
    ax2.plot(i, b, 'g+')
    ax.plot(i, loss, 'ro')
  
ax1.set_title('Weight over iterations')
ax1.set_xlabel('iterations')
ax1.set_ylabel('Weight')
  
ax2.set_title('Bias over iterations')
ax2.set_xlabel('iterations')
ax2.set_ylabel('Bias')
  
ax.set_title('Losses over iterations')
ax.set_xlabel('iterations')
ax.set_ylabel('Losses')
  
plt.show()

print('Weight :',W)
print('Bias :',b)
  
plt.scatter(x, y)
plt.plot(x, W * x + b, color='red')
plt.title('Regression Line')
plt.xlabel('Input')
plt.ylabel('Target')
plt.show()