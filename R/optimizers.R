adagrad = function(lrate = 0.01)
{
    list(opt_name = "adagrad",
         lrate = lrate)
}

rmsprop = function(lrate = 0.0001, decay = 0.99)
{
    list(opt_name = "rmsprop",
         lrate = lrate,
         decay = decay)
}

adam = function(lrate = 0.001, b1 = 0.9, b2 = 0.999, b1_t = 0.9, b2_t = 0.999)
{
    list(opt_name = "adam",
         lrate = lrate,
         b1 = b1,
         b2 = b2,
         b1_t = b1_t,
         b2_t = b2_t)
}

sgd = function(lrate = 0.01, decay = 0)
{
    list(opt_name = "sgd",
         lrate = lrate,
         decay = decay)
}

momentum = function(lrate = 0.01, decay = 0, momentum = 0.9)
{
    list(opt_name = "momentum",
         lrate = lrate,
         decay = decay,
         momentum = momentum)
}
