activation_id = c(identity   = 0L,
                  sigmoid    = 1L,
                  relu       = 2L,
                  leaky_relu = 3L,
                  elu        = 4L,
                  softmax    = 5L,
                  tan_h      = 6L,
                  tan_hp1m2  = 7L)

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

fc = layer_fully_connected
