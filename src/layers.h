#ifndef TINYDNN_LAYERS_H
#define TINYDNN_LAYERS_H

#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>

void add_layer_fully_connected(
    tiny_dnn::network<tiny_dnn::sequential>* net, Rcpp::List layer
);

void add_layer_convolutional(
    tiny_dnn::network<tiny_dnn::sequential>* net, Rcpp::List layer
);

void add_layer_average_pooling(
    tiny_dnn::network<tiny_dnn::sequential>* net, Rcpp::List layer
);

#endif // TINYDNN_LAYERS_H
