library(keras)
library(tidyverse)

# Parameters --------------------------------------------------------------
batch_size <- 100
epochs <- 50

# Data Preparation --------------------------------------------------------

# See ?dataset_cifar10 for more info
cifar10 <- dataset_cifar10()

# Feature scale RGB values in test and train inputs  
x_train <- cifar10$train$x/255
x_test <- cifar10$test$x/255
y_train <- to_categorical(cifar10$train$y, num_classes = 10)
y_test <- to_categorical(cifar10$test$y, num_classes = 10)
img_rows <- 32
img_cols <- 32
input_shape <- c(img_rows, img_cols, 3)

# Define Model -----------------------------------------------------------
# https://github.com/Coderx7/SimpNet/blob/master/SimpNetV2/Logs/CIFAR10/caffe_8.9M_95.89.log
# "CIFAR10_SimpleNet_GP_13L_drpall_8Mil_66_DRP_After_Pooling"

normal_krnl <- initializer_random_normal(stddev = 0.01)
model <-  keras_model_sequential() %>%
  ### Block 1
  ## Conv 1_0 (conv 1)
  layer_conv_2d(filters = 128, kernel_size = c(3,3), padding = "same",
                kernel_initializer = "glorot_uniform", # aka Xavier
                input_shape = input_shape) %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 2
  ## Conv 2_0 (conv 2)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same") %>% 
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 2_1 (conv 3)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same",
                kernel_initializer = normal_krnl) %>% # gaussian Kernel
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 2_2 (conv 4)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same",
                kernel_initializer = normal_krnl) %>% # Gaussian
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 3
  ## Conv 3_0 (conv 5)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same",
                kernel_initializer = normal_krnl) %>% # Gaussian
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 4
  ## Conv 4_0 (conv 6)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 4_1 (conv 7)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 4_3 (conv 8)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  ## Conv 4_4 (conv 9)
  layer_conv_2d(filters = 182, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 5
  ## Conv 5_0 (conv 10)
  layer_conv_2d(filters = 430, kernel_size = c(3,3), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2), stride = 2) %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 6
  ## Conv 6_0 (conv 11)
  layer_conv_2d(filters = 430, kernel_size = c(1,1), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  
  ### Block 7
  ## conv 7_0 (conv 12)
  layer_conv_2d(filters = 455, kernel_size = c(1,1), padding = "same") %>%
  layer_batch_normalization(momentum = 0.95, scale = TRUE) %>% 
  layer_activation("relu") %>%
  layer_dropout(rate = 0.2) %>%
  
  ### Block 8
  ## Conv 8_0 (conv 13)
  layer_conv_2d(filters = 600, kernel_size = c(3,3), padding = "same") %>%
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

# Paperspace after 55 epochs
# > cat('Test loss:', scores[[1]], '\n')
# Test loss: 0.5701714 
# > cat('Test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.887 
> 
