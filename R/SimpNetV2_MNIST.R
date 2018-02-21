library(keras)
library(tidyverse)
# Data Preparation -----------------------------------------------------

batch_size <- 100
num_classes <- 10
epochs <- 5

# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Redefine  dimension of train/test inputs
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define Model -----------------------------------------------------------

# SimpNetV2
# https://github.com/Coderx7/SimpNet/blob/master/SimpNetV2/Logs/MNIST/caffe_99.75.log
# "MNIST_SimpleNet_GP_13L_drpall_5Mil_66_maxdrp"

normal_krnl <- initializer_random_normal(stddev = 0.01)
model <-  keras_model_sequential() %>%
  ### Block 1
  ## Conv 1_0 (conv 1)
  layer_conv_2d(filters = 66, kernel_size = c(3,3), padding = "same",
                kernel_initializer = "glorot_uniform", # aka Xavier
                input_shape = input_shape) %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 2
  ## Conv 2_0 (conv 2)
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same") %>% 
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 2_1 (conv 3)
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same",
                kernel_initializer = normal_krnl) %>% # gaussian Kernel
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 2_2 (conv 4)
  layer_conv_2d(filters = 64, kernel_size = c(3,3), padding = "same",
                kernel_initializer = normal_krnl) %>% # Gaussian
  layer_batch_normalization(momentum = 0.95, 
                            scale = TRUE, gamma_regularizer = "l2") %>% # <- ??
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 3
  ## Conv 3_0 (conv 5)
  layer_conv_2d(filters = 96, kernel_size = c(3,3), padding = "same",
                kernel_initializer = normal_krnl) %>% # Gaussian
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 4
  ## Conv 4_0 (conv 6)
  layer_conv_2d(filters = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 4_1 (conv 7)
  layer_conv_2d(filters = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 4_3 (conv 8)
  layer_conv_2d(filters = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 4_4 (conv 9)
  layer_conv_2d(filters = 96, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%

  ### Block 5
  ## Conv 5_0 (conv 10)
  layer_conv_2d(filters = 144, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 6
  ## Conv 6_0 (conv 11)
  layer_conv_2d(filters = 144, kernel_size = c(1,1), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  
  ### Block 7
  ## conv 7_0 (conv 12)
  layer_conv_2d(filters = 178, kernel_size = c(1,1), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 8
  ## Conv 8_0 (conv 13)
  layer_conv_2d(filters = 216, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>%
  layer_activation("relu") %>%
  layer_global_max_pooling_2d() %>%
  layer_dropout(rate = 0.2) %>%
  ### Inference
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)
# Train model
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2
)
scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')

### on paperspace, 10 epochs
# > cat('Test loss:', scores[[1]], '\n')
# Test loss: 0.01990014 
# > cat('Test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.9949 