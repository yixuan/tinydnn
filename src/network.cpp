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



// [[Rcpp::export]]
SEXP net_seq_fit(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
                 Rcpp::NumericMatrix x,
                 Rcpp::NumericVector y,
                 int batch_size,
                 int epochs,
                 std::string optimizer)
{
    using namespace tiny_dnn;

    int n = x.nrow();
    int p = x.ncol();

    std::vector<vec_t> input;
    std::vector<vec_t> output;

    input.reserve(n);
    output.reserve(n);

    for(int i = 0; i < n; i++)
    {
        vec_t row(p);
        for(int j = 0; j < p; j++)
        {
            row[j] = x(i, j);
        }
        input.push_back(row);
        output.push_back(vec_t(1, y[i]));
    }

    adagrad opt;

    net->fit<mse>(opt, input, output, batch_size, epochs);

    return R_NilValue;
}
