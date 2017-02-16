NetworkSequential = setRefClass(
    "NetworkSequential",
    fields = list(net = "externalptr",
                  type = "character",
                  levels = "character")
)

NetworkSequential$methods(
    name = function()
    {
        "Name of the network"

        net_seq_name(.self$net)
    },

    add_layer = function(layer)
    {
        "Add one layer to the network"

        net_seq_add_layer(.self$net, layer)
        invisible(.self)
    },

    layer_size = function()
    {
        "Number of layers"

        net_seq_layer_size(.self$net)
    },

    out_data_size = function()
    {
        "Total number of elements of output data"

        if(!.self$layer_size())  return(0L)
        net_seq_out_data_size(.self$net)
    },

    in_data_size = function()
    {
        "Total number of elements of input data"

        if(!.self$layer_size())  return(0L)
        net_seq_in_data_size(.self$net)
    },

    load = function(filename)
    {
        ## TODO
    },

    save = function(filename)
    {
        ## TODO
    },

    to_json = function()
    {
        ## TODO
    },

    from_json = function(json_string)
    {
        ## TODO
    },

    fit = function(x, y, batch_size, epochs = 10,
                   optimizer = adagrad(), verbose = TRUE)
    {
        "Fitting a regression or classification model"

        ## Type checks
        if(!is.numeric(x))
            stop("'x' must be a numeric matrix or vector")
        if((!is.numeric(y)) && (!is.factor(y)) && (!is.character(y)))
            stop("'y' must be a numeric matrix/vector, or a factor/character vector")
        if(is.character(y))
            y = factor(y)
        if(is.factor(y) && is.matrix(y))
            stop("'y' must be a numeric matrix/vector, or a factor/character vector")

        batch_size = as.integer(batch_size)
        epochs = as.integer(epochs)
        optimizer = as.list(optimizer)
        verbose = as.logical(verbose)

        ## Force x to be a matrix
        if(!is.matrix(x))
            dim(x) = c(length(x), 1L)

        ## Numeric y -- regression
        if(is.numeric(y))
        {
            ## Used to let predict() know the prediction type
            .self$type = "regression"

            ## Force y to be a matrix
            if(!is.matrix(y))  dim(y) = c(length(y), 1L)

            ## Check number of observations
            if(nrow(x) != nrow(y))
                stop("'x' and 'y' have different number of observations")

            ## Check the compatibility of dimensionality with the network
            dimx = ncol(x)
            dimy = ncol(y)
            if(.self$in_data_size() != dimx)
                stop(sprintf("mismatch of dimensionality between network and data\nnetwork input: %d variable(s)\ndata input: %d variable(s)",
                     .self$in_data_size(), dimx))
            if(.self$out_data_size() != dimy)
                stop(sprintf("mismatch of dimensionality between network and data\nnetwork output: %d variable(s)\ndata output: %d variable(s)",
                     .self$out_data_size(), dimy))

            ## Call model fitting function
            net_seq_regression_fit(.self$net, x, y, batch_size, epochs, optimizer, verbose)
        } else {
            ## Factor y -- Classification

            ## Used to let predict() know the prediction type and the class levels
            .self$type = "classification"
            .self$levels = levels(y)

            ## Convert factor to integer, 0-indexed
            ylabel = as.integer(y) - 1L
            n = length(y)

            ## Check number of observations
            if(nrow(x) != n)
                stop("'x' and 'y' have different number of observations")

            ## Check the compatibility of dimensionality with the network
            dimx = ncol(x)
            dimy = nlevels(y)
            if(.self$in_data_size() != dimx)
                stop(sprintf("mismatch of dimensionality between network and data\nnetwork input: %d variable(s)\ndata input: %d variable(s)",
                             .self$in_data_size(), dimx))
            if(.self$out_data_size() != dimy)
                stop(sprintf("mismatch of dimensionality between network and data\nnetwork output: %d variable(s)\ndata output: %d level(s)",
                             .self$out_data_size(), dimy))

            ## Call model fitting function
            net_seq_classification_fit(.self$net, x, ylabel, batch_size, epochs, optimizer, verbose)
        }

        invisible(.self)
    },

    predict = function(newx, type = c("class", "prob"))
    {
        "Predicting new observations based on the fitted model"

        ## Type check
        if(!is.numeric(newx))
            stop("'newx' must be a numeric matrix or vector")

        ## Force newx to be a matrix
        if(!is.matrix(newx))
            dim(newx) = c(length(newx), 1L)

        ## Check the compatibility of dimensionality with the network
        dimx = ncol(newx)
        if(.self$in_data_size() != dimx)
            stop(sprintf("mismatch of dimensionality between network and data\nnetwork input: %d variable(s)\ndata input: %d variable(s)",
                         .self$in_data_size(), dimx))

        out_type = match.arg(type)
        if(.self$type == "classification" && out_type == "class")
        {
            res = net_seq_classification_predict(.self$net, newx)
            res = factor(.self$levels[res + 1], levels = .self$levels)
        } else {
            ## This includes regression, and classification with class = "prob"

            res = net_seq_regression_predict(.self$net, newx)
            ## If the result has only one column, make it a vector
            if(ncol(res) == 1)  dim(res) = NULL
        }

        res
    }
)

net_seq = function(name = "")
{
    NetworkSequential$new(net = net_seq_constructor(name))
}

if(!isGeneric("%<%"))
    setGeneric("%<%", function(e1, e2) standardGeneric("%<%"))

setMethod("%<%", signature(e1 = "NetworkSequential", e2 = "list"),
          function(e1, e2) {
              e1$add_layer(e2)
              invisible(e1)
          }
)
