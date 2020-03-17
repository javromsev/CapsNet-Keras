## This class is aimed to preprocess CIFAR10/CIFAR100 dataset (available on Keras API) in order to convert all training sets 
## from Tensors to Array

def create_cifar_array():
  
  # Import dataset from KERAS API
  
  from keras.datasets import cifar100
  
  # Define train and test sets (still RGB format)
  
  (rgb_x_train, y_train), (rgb_x_test, y_test) = cifar100.load_data(label_mode='fine')
  
  
  # Convert from RGB format to grayscale
  
  x_train_tens = tf.image.rgb_to_grayscale(rgb_x_train, name=None)
  x_test_tens = tf.image.rgb_to_grayscale(rgb_x_test, name=None)
  
  # Convert Tensors to array
  
  x_train = x_train_tens.eval(session=tf.compat.v1.Session())
  x_test = x_test_tens.eval(session=tf.compat.v1.Session())
  
  return x_train,y_train,x_test,y_test



