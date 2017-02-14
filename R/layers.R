activation_id = c(identity   = 0L,
                  sigmoid    = 1L,
                  relu       = 2L,
                  leaky_relu = 3L,
                  elu        = 4L,
                  softmax    = 5L,
                  tan_h      = 6L,
                  tan_hp1m2  = 7L)

##' Layers to Build Deep Neural Networks
##'
##' Various layers that can be combined to build a deep neural network.
##'
##' @rdname layers
##'
##' @param in_dim Number of elements in the input
##' @param out_dim Number of elements in the output
##' @param activation Activation function applied to this layer. See section
##'                   \strong{Activation Functions} for details.
##' @param has_bias Whether to include the bias element
##'
##' @section List of Layers:
##' Currently the following layers are supported:
##' \itemize{
##'   \item Fully-connected layer: \code{layer_fully_connected()}, or \code{fc()} for short
##'   \item Convolutoinal layer: \code{layer_convolutional()}, \code{conv()}
##' }
##' More types of layers are to be added.
##'
##' @section Activation Functions:
##' Currently the following activation functions are supported:
##'
##' \itemize{
##'   \item identity
##'   \item sigmoid
##'   \item relu
##'   \item leaky_relu
##'   \item elu
##'   \item softmax
##'   \item tan_h
##'   \item tan_hp1m2
##' }
##'
##' @export
##'
layer_fully_connected = function(
    in_dim,
    out_dim,
    activation = "sigmoid",
    has_bias = TRUE
)
{
    list(layer_id = 0L,
         act_id   = activation_id[activation],
         in_dim   = as.integer(in_dim),
         out_dim  = as.integer(out_dim),
         has_bias = as.logical(has_bias))
}

##' @rdname layers
##' @export
fc = layer_fully_connected

##' @rdname layers
##' @param in_width Input image width
##' @param in_height Input image height
##' @param window_width Window width of convolution
##' @param window_height Window height of convolution
##' @param in_channels Input image channels
##' @param out_channels Output image channels
##' @param pad_type Rounding strategy
##' @param w_stride The horizontal interval at which to apply the filters
##' @param h_stride The vertical interval at which to apply the filters
layer_convolutional = function(
    in_width, in_height, window_width, window_height, in_channels, out_channels,
    activation = "sigmoid",
    pad_type = c("valid", "same"), has_bias = TRUE, w_stride = 1L, h_stride = 1L
)
{
    list(layer_id      = 1L,
         act_id        = activation_id[activation],
         in_width      = as.integer(in_width),
         in_height     = as.integer(in_height),
         window_width  = as.integer(window_width),
         window_height = as.integer(window_height),
         in_channels   = as.integer(in_channels),
         out_channels  = as.integer(out_channels),
         has_bias      = as.logical(has_bias),
         w_stride      = as.integer(w_stride),
         h_stride      = as.integer(h_stride))
}

##' @rdname layers
##' @export
conv = layer_convolutional
