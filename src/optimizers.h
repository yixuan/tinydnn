#ifndef TINYDNN_OPTIMIZERS_H
#define TINYDNN_OPTIMIZERS_H

#include <memory>
#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>


std::shared_ptr<tiny_dnn::optimizer> get_optimizer(Rcpp::List opt);


#endif // TINYDNN_OPTIMIZERS_H
