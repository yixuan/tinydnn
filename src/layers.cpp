#include "layers.h"

enum {
    ACT_IDENTITY = 0,
    ACT_SIGMOID,
    ACT_RELU,
    ACT_LEAKY_RELU,
    ACT_ELU,
    ACT_SOFTMAX,
    ACT_TAN_H,
    ACT_TAN_HP1M2
};

void add_layer_fully_connected(tiny_dnn::network<tiny_dnn::sequential>* net,
                               Rcpp::List layer)
{
    using namespace tiny_dnn;
    using namespace tiny_dnn::activation;
    using tiny_dnn::serial_size_t;

    int act_id = layer["act_id"];
    serial_size_t in_dim = Rcpp::as<serial_size_t>(layer["in_dim"]);
    serial_size_t out_dim = Rcpp::as<serial_size_t>(layer["out_dim"]);
    bool has_bias = Rcpp::as<bool>(layer["has_bias"]);

    switch(act_id)
    {
    case ACT_IDENTITY:
        (*net) << fully_connected_layer<identity>(in_dim, out_dim, has_bias);
        break;
    case ACT_SIGMOID:
        (*net) << fully_connected_layer<sigmoid>(in_dim, out_dim, has_bias);
        break;
    case ACT_RELU:
        (*net) << fully_connected_layer<relu>(in_dim, out_dim, has_bias);
        break;
    case ACT_LEAKY_RELU:
        (*net) << fully_connected_layer<leaky_relu>(in_dim, out_dim, has_bias);
        break;
    case ACT_ELU:
        (*net) << fully_connected_layer<elu>(in_dim, out_dim, has_bias);
        break;
    case ACT_SOFTMAX:
        (*net) << fully_connected_layer<softmax>(in_dim, out_dim, has_bias);
        break;
    case ACT_TAN_H:
        (*net) << fully_connected_layer<tan_h>(in_dim, out_dim, has_bias);
        break;
    case ACT_TAN_HP1M2:
        (*net) << fully_connected_layer<tan_hp1m2>(in_dim, out_dim, has_bias);
        break;
    default:
        Rcpp::stop("unsupported activation function");
    }
}
