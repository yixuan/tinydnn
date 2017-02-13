#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>
#include "layers.h"

// [[Rcpp::export]]
SEXP net_seq_regression_fit(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x,
    Rcpp::NumericVector y,
    int batch_size,
    int epochs,
    std::string optimizer
)
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



// [[Rcpp::export]]
Rcpp::NumericVector net_seq_regression_predict(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x
)
{
    using namespace tiny_dnn;

    int n = x.nrow();
    int p = x.ncol();

    Rcpp::NumericVector yhat(n);

    for(int i = 0; i < n; i++)
    {
        vec_t row(p);
        for(int j = 0; j < p; j++)
        {
            row[j] = x(i, j);
        }
        vec_t res = net->predict(row);
        yhat[i] = res[0];
    }

    return yhat;
}
