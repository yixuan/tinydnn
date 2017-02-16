#include "optimizers.h"

using namespace tiny_dnn;
using std::shared_ptr;

std::shared_ptr<tiny_dnn::optimizer> get_optimizer(Rcpp::List opt)
{
    std::string opt_name = Rcpp::as<std::string>(opt["opt_name"]);

    // Use Adagrad as a default method
    shared_ptr<adagrad> opt_ptr = std::make_shared<adagrad>();
    opt_ptr->alpha = Rcpp::as<float_t>(opt["lrate"]);

    return opt_ptr;
}
