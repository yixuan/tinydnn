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

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using tiny_dnn::serial_size_t;

// Fully-connected layer
void add_layer_fully_connected(
    tiny_dnn::network<tiny_dnn::sequential>* net, Rcpp::List layer
)
{
    const int act_id = layer["act_id"];
    const serial_size_t in_dim = Rcpp::as<serial_size_t>(layer["in_dim"]);
    const serial_size_t out_dim = Rcpp::as<serial_size_t>(layer["out_dim"]);
    const bool has_bias = Rcpp::as<bool>(layer["has_bias"]);

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

// Convolutional layer
void add_layer_convolutional(
    tiny_dnn::network<tiny_dnn::sequential>* net, Rcpp::List layer
)
{
    const int act_id = layer["act_id"];
    const serial_size_t in_width = Rcpp::as<serial_size_t>(layer["in_width"]);
    const serial_size_t in_height = Rcpp::as<serial_size_t>(layer["in_height"]);
    const serial_size_t window_width = Rcpp::as<serial_size_t>(layer["window_width"]);
    const serial_size_t window_height = Rcpp::as<serial_size_t>(layer["window_height"]);
    const serial_size_t in_channels = Rcpp::as<serial_size_t>(layer["in_channels"]);
    const serial_size_t out_channels = Rcpp::as<serial_size_t>(layer["out_channels"]);
    const padding pad_type = (Rcpp::as<std::string>(layer["pad_type"]) == "same") ?
                              padding::same :
                              padding::valid;
    const bool has_bias = Rcpp::as<bool>(layer["has_bias"]);
    const serial_size_t w_stride = Rcpp::as<serial_size_t>(layer["w_stride"]);
    const serial_size_t h_stride = Rcpp::as<serial_size_t>(layer["h_stride"]);

    switch(act_id)
    {
    case ACT_IDENTITY:
        (*net) << convolutional_layer<identity>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_SIGMOID:
        (*net) << convolutional_layer<sigmoid>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_RELU:
        (*net) << convolutional_layer<relu>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_LEAKY_RELU:
        (*net) << convolutional_layer<leaky_relu>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_ELU:
        (*net) << convolutional_layer<elu>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_SOFTMAX:
        (*net) << convolutional_layer<softmax>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_TAN_H:
        (*net) << convolutional_layer<tan_h>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    case ACT_TAN_HP1M2:
        (*net) << convolutional_layer<tan_hp1m2>(
                      in_width, in_height, window_width, window_height,
                      in_channels, out_channels, pad_type, has_bias, w_stride, h_stride
                  );
        break;
    default:
        Rcpp::stop("unsupported activation function");
    }
}
