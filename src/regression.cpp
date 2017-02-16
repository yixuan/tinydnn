#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>
#include "layers.h"
#include "optimizers.h"

// [[Rcpp::export]]
SEXP net_seq_regression_fit(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x,
    Rcpp::NumericMatrix y,
    int batch_size,
    int epochs,
    Rcpp::List opt,
    bool verbose
)
{
    using namespace tiny_dnn;

    const int n = x.nrow();
    const int px = x.ncol();
    const int py = y.ncol();

    std::vector<vec_t> input;
    std::vector<vec_t> output;

    input.reserve(n);
    output.reserve(n);

    // Copy data
    vec_t rowx(px);
    vec_t rowy(py);
    for(int i = 0; i < n; i++)
    {
        // Fill input
        for(int j = 0; j < px; j++)
        {
            rowx[j] = x(i, j);
        }
        input.push_back(rowx);

        // Fill output
        if(py == 1)
        {
            rowy[0] = y[i];
        } else {
            for(int j = 0; j < py; j++)
            {
                rowy[j] = y(i, j);
            }
        }
        output.push_back(rowy);
    }

    std::shared_ptr<tiny_dnn::optimizer> opt_ptr = get_optimizer(opt);

    timer t;
    int epoch = 0;

    net->fit<mse>(*opt_ptr, input, output, batch_size, epochs,
        // called for each mini-batch
        []() {

        },
        // called for each epoch
        [verbose, &t, &epoch]() {
            if(verbose)
            {
                Rcpp::Rcout << "[Epoch " << epoch << "]: " << t.elapsed() << "s" << std::endl;
                t.restart();
                epoch++;
            }
        }
    );

    return R_NilValue;
}



// [[Rcpp::export]]
Rcpp::NumericMatrix net_seq_regression_predict(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x
)
{
    using namespace tiny_dnn;

    const int n = x.nrow();
    const int px = x.ncol();
    const int py = net->out_data_size();

    Rcpp::NumericMatrix pred(n, py);
    vec_t rowx(px);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < px; j++)
        {
            rowx[j] = x(i, j);
        }

        vec_t rowy = net->predict(rowx);
        if(py == 1)
        {
            pred[i] = rowy[0];
        } else {
            for(int j = 0; j < py; j++)
            {
                pred(i, j) = rowy[j];
            }
        }
    }

    return pred;
}
