#include "optimizers.h"

using namespace tiny_dnn;
using std::shared_ptr;

std::shared_ptr<tiny_dnn::optimizer> get_optimizer(Rcpp::List opt)
{
    std::string opt_name = Rcpp::as<std::string>(opt["opt_name"]);

    if(opt_name == "rmsprop")
    {
        // RMSprop
        shared_ptr<RMSprop> opt_ptr = std::make_shared<RMSprop>();
        opt_ptr->alpha = Rcpp::as<float_t>(opt["lrate"]);
        opt_ptr->mu = Rcpp::as<float_t>(opt["decay"]);

        return opt_ptr;
    } else if(opt_name == "adam") {
        // Adam
        shared_ptr<adam> opt_ptr = std::make_shared<adam>();
        opt_ptr->alpha = Rcpp::as<float_t>(opt["lrate"]);
        opt_ptr->b1 = Rcpp::as<float_t>(opt["b1"]);
        opt_ptr->b2 = Rcpp::as<float_t>(opt["b2"]);
        opt_ptr->b1_t = Rcpp::as<float_t>(opt["b1_t"]);
        opt_ptr->b2_t = Rcpp::as<float_t>(opt["b2_t"]);

        return opt_ptr;
    } else if(opt_name == "sgd") {
        // Stochastic gradient descent without momentum
        shared_ptr<gradient_descent> opt_ptr = std::make_shared<gradient_descent>();
        opt_ptr->alpha = Rcpp::as<float_t>(opt["lrate"]);
        opt_ptr->lambda = Rcpp::as<float_t>(opt["decay"]);

        return opt_ptr;
    } else if(opt_name == "momentum") {
        // Stochastic gradient descent with momentum
        shared_ptr<momentum> opt_ptr = std::make_shared<momentum>();
        opt_ptr->alpha = Rcpp::as<float_t>(opt["lrate"]);
        opt_ptr->lambda = Rcpp::as<float_t>(opt["decay"]);
        opt_ptr->mu = Rcpp::as<float_t>(opt["momentum"]);

        return opt_ptr;
    }

    // Use Adagrad as a default method
    shared_ptr<adagrad> opt_ptr = std::make_shared<adagrad>();
    opt_ptr->alpha = Rcpp::as<float_t>(opt["lrate"]);

    return opt_ptr;
}
