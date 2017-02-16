## tinydnn

### Introduction

**tinydnn** is an (experimental) R wrapper of the
[tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) library for implementing
Deep Neural Networks (DNN). The largest advantage of `tiny-dnn` over other deep
learning frameworks is its minimal dependency on external software and the
ease of installation. As a result, the R package **tinydnn** is also very
convenient to install as long as you have a C++ 11 compiler, and it runs on
all major platforms including Linux, Mac, Windows etc.

**tinydnn** may be a good option for building DNN models if:

- You use R! (You may want to consider [MXNet](https://github.com/dmlc/mxnet) first)
- You have a CPU-only environment with limited resources
- You want to quickly try DNN models without spending too much time on
installation and configuration
- You need different packages to compare the results
- You want to learn the internals of DNN (The included `tiny-dnn` library
provides an excellent coding example of DNN)

### Development Status

**tinydnn** is still in the experiment stage. Functions and interface may change,
and more features will be added per request. Feedbacks and contributions are
highly welcome.

### Example

This package has not been fully documented. The examples below are mostly
self-explanatory.

#### Regression

We use the [wine quality data](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
on UCI machine learning repository to demonstrate a regression example, in which
we use several attributes of the wine to predict its quality.

```r
## Wine quality data set
## https://archive.ics.uci.edu/ml/datasets/Wine+Quality
dat_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
dat = read.csv(url(dat_url), sep = ";")
n = nrow(dat)
x = scale(as.matrix(dat[, -ncol(dat)]))
y = dat[, ncol(dat)]

## Splitting training and testing data
set.seed(123)
ind = sample(1:n, floor(0.8 * n))
train_x = x[ind, ]
train_y = y[ind]
test_x = x[-ind, ]
test_y = y[-ind]

## Create a neural network
library(tinydnn)
net = net_seq()

## Add layers
# A fully-connected layer that takes the data as input,
# with 20 hidden units and the ReLU activation function
net$add_layer(fc(ncol(train_x), 20, act = "relu"))
# A second layer with 20 hidden units                                      
net$add_layer(fc(20, 20, act = "relu"))
# The output layer
net$add_layer(fc(20, 1, act = "identity"))

# There is also a "%<%" operator that can be used to build the network
net = net_seq()
net %<%
    fc(ncol(train_x), 20, act = "relu") %<%
    fc(20, 20, act = "relu") %<%
    fc(20, 1, act = "identity")

## Fit the model on the data set
net$fit(train_x, train_y, batch_size = 100, epochs = 100, verbose = TRUE)

## Make prediction
pred_train = net$predict(train_x)
pred_test = net$predict(test_x)

mean((train_y - pred_train)^2)
mean((test_y - pred_test)^2)
```

#### Classification

Since the quality of wine is coded as an integer from 1 to 10 (actually 3 to 8
in this data set), we can also regard this as a classification problem. The
code below shows how we build a neural network for classification.

```r
## Make the wine quality a categorical variable
train_y = factor(train_y)
test_y = factor(test_y)

## Construct the network
net = net_seq()
net %<%
    fc(ncol(train_x), 20, act = "sigmoid") %<%
    fc(20, 30, act = "sigmoid") %<%
    fc(30, 20, act = "sigmoid") %<%
    fc(20, nlevels(train_y), act = "softmax")

net$fit(train_x, train_y, batch_size = 100, epochs = 100, verbose = TRUE)
pred_train = net$predict(train_x, type = "class")
pred_test = net$predict(test_x, type = "class")

## Confusion matrix
table(pred_train, train_y)
table(pred_test, test_y)

## If class probabilities are required, use the `type = "prob"` option
prob = net$predict(test_x, type = "prob")
```

In the examples above we only use fully-connected layers to construct the
network. There are other types of layers supported by **tinydnn**, for example
convolutional layers. See `?layers` for a list of currently supported ones.

### TODO

- Random seed. If possible use the RNG provided by R itself.
- Add more layers implemented by the `tiny-dnn` library.
- Add convenient functions to manipulate networks and layers.
- Support different optimization methods.
