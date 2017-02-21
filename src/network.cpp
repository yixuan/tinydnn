#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>
#include "layers.h"
#include "utils.h"

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
    case 1:
        add_layer_convolutional(net, layer);
        break;
    case 2:
        add_layer_average_pooling(net, layer);
        break;
    case 3:
        add_layer_max_pooling(net, layer);
        break;
    default:
        Rcpp::stop("unimplemented layer type");
    }

    return R_NilValue;
}

// [[Rcpp::export]]
Rcpp::List net_seq_get_weights(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net)
{
    using tiny_dnn::vec_t;
    using tiny_dnn::layer;

    int layer_size = net->layer_size();
    Rcpp::List res(layer_size);

    for(int i = 0; i < layer_size; i++)
    {
        const layer* net_layer = (*net)[i];
        std::string type = net_layer->layer_type();
        std::vector<const vec_t*> params = net_layer->weights();
        int nparam = params.size();

        // For fully-connected layers, output type, weights, and bias
        if(type == "fully-connected")
        {
            Rcpp::List lst = Rcpp::List::create(
                Rcpp::Named("type") = type,
                Rcpp::Named("weights") = vec_t_to_rcpp_matrix(
                    params[0],
                    net_layer->out_data_size(),
                    net_layer->in_data_size()
                ),
                Rcpp::Named("bias") = vec_t_to_rcpp_vector(params[1])
            );
            res[i] = lst;
        } else {
            // (Currently) for other layers, output general parameters
            Rcpp::List lst(nparam + 1);
            lst[0] = type;
            for(int j = 0; j < nparam; j++)
            {
                lst[j + 1] = vec_t_to_rcpp_vector(params[j]);
            }
            res[i] = lst;
        }
    }

    return res;
}

