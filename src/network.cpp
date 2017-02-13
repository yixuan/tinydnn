#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>
#include "layers.h"

// [[Rcpp::export]]
Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net_seq_constructor(std::string name)
{
    using namespace tiny_dnn;

    network<sequential>* net = new network<sequential>(name);

    return Rcpp::XPtr< network<sequential> >(net);
}

// [[Rcpp::export]]
std::string net_seq_name(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net)
{
    return net->name();
}

// [[Rcpp::export]]
int net_seq_layer_size(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net)
{
    return net->layer_size();
}

// [[Rcpp::export]]
int net_seq_out_data_size(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net)
{
    return net->out_data_size();
}

// [[Rcpp::export]]
int net_seq_in_data_size(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net)
{
    return net->in_data_size();
}



// [[Rcpp::export]]
SEXP net_seq_add_layer(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
                       Rcpp::List layer)
{
    int id = layer["layer_id"];
    switch(id)
    {
    case 0:
        add_layer_fully_connected(net, layer);
        break;
    default:
        Rcpp::stop("unimplemented layer type");
    }

    return R_NilValue;
}
