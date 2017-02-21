#ifndef TINYDNN_UTILS_H
#define TINYDNN_UTILS_H

#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>

inline Rcpp::NumericVector vec_t_to_rcpp_vector(const tiny_dnn::vec_t* v)
{
    Rcpp::NumericVector res(v->size());
    std::copy(v->begin(), v->end(), res.begin());

    return res;
}

inline Rcpp::NumericMatrix vec_t_to_rcpp_matrix(const tiny_dnn::vec_t* v, int m, int n)
{
    Rcpp::NumericMatrix res(m, n);
    std::copy(v->begin(), v->end(), res.begin());

    return res;
}


#endif // TINYDNN_UTILS_H
