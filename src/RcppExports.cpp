// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "../inst/include/tinydnn.h"
#include <Rcpp.h>

using namespace Rcpp;

// net_seq_classification_fit
SEXP net_seq_classification_fit(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net, Rcpp::NumericMatrix x, Rcpp::IntegerVector y, int batch_size, int epochs, Rcpp::List opt, bool verbose);
RcppExport SEXP tinydnn_net_seq_classification_fit(SEXP netSEXP, SEXP xSEXP, SEXP ySEXP, SEXP batch_sizeSEXP, SEXP epochsSEXP, SEXP optSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type opt(optSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_classification_fit(net, x, y, batch_size, epochs, opt, verbose));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_classification_predict
Rcpp::IntegerVector net_seq_classification_predict(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net, Rcpp::NumericMatrix x);
RcppExport SEXP tinydnn_net_seq_classification_predict(SEXP netSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_classification_predict(net, x));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_constructor
Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net_seq_constructor(std::string name);
RcppExport SEXP tinydnn_net_seq_constructor(SEXP nameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< std::string >::type name(nameSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_constructor(name));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_name
std::string net_seq_name(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net);
RcppExport SEXP tinydnn_net_seq_name(SEXP netSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_name(net));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_layer_size
int net_seq_layer_size(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net);
RcppExport SEXP tinydnn_net_seq_layer_size(SEXP netSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_layer_size(net));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_out_data_size
int net_seq_out_data_size(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net);
RcppExport SEXP tinydnn_net_seq_out_data_size(SEXP netSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_out_data_size(net));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_in_data_size
int net_seq_in_data_size(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net);
RcppExport SEXP tinydnn_net_seq_in_data_size(SEXP netSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_in_data_size(net));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_add_layer
SEXP net_seq_add_layer(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net, Rcpp::List layer);
RcppExport SEXP tinydnn_net_seq_add_layer(SEXP netSEXP, SEXP layerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type layer(layerSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_add_layer(net, layer));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_regression_fit
SEXP net_seq_regression_fit(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net, Rcpp::NumericMatrix x, Rcpp::NumericMatrix y, int batch_size, int epochs, Rcpp::List opt, bool verbose);
RcppExport SEXP tinydnn_net_seq_regression_fit(SEXP netSEXP, SEXP xSEXP, SEXP ySEXP, SEXP batch_sizeSEXP, SEXP epochsSEXP, SEXP optSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type epochs(epochsSEXP);
    Rcpp::traits::input_parameter< Rcpp::List >::type opt(optSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_regression_fit(net, x, y, batch_size, epochs, opt, verbose));
    return rcpp_result_gen;
END_RCPP
}
// net_seq_regression_predict
Rcpp::NumericMatrix net_seq_regression_predict(Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net, Rcpp::NumericMatrix x);
RcppExport SEXP tinydnn_net_seq_regression_predict(SEXP netSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > >::type net(netSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(net_seq_regression_predict(net, x));
    return rcpp_result_gen;
END_RCPP
}
