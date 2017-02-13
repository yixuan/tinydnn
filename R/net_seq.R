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
    }
)

net_seq = function(name = "")
{
    NetworkSequential$new(net = net_seq_constructor(name))
}
