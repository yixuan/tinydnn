#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net_seq()
{
    using namespace tiny_dnn;

    network<sequential>* net = new network<sequential>();

    return Rcpp::XPtr< network<sequential> >(net);
}
