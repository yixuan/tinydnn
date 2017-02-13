NetworkSequential = setRefClass(
    "NetworkSequential",
    fields = list(net = "externalptr")
)

NetworkSequential$methods(
    name = function()
    {
        "Name of the network"

        net_seq_name(.self$net)
    },

    layer_size = function()
    {
        "Number of layers"

        net_seq_layer_size(.self$net)
    },

    out_data_size = function()
    {
        "Total number of elements of output data"

        net_seq_out_data_size(.self$net)
    },

    in_data_size = function()
    {
        "Total number of elements of input data"

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
    }
)

net_seq = function(name = "")
{
    NetworkSequential$new(net = net_seq_constructor(name))
}
