#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>
#include "layers.h"
#include "optimizers.h"

// [[Rcpp::export]]
SEXP net_seq_classification_fit(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x,
    Rcpp::IntegerVector y,
    int batch_size,
    int epochs,
    Rcpp::List opt,
    bool verbose
)
{
    using namespace tiny_dnn;

    const int n = x.nrow();
    const int p = x.ncol();

    std::vector<vec_t> input;
    std::vector<label_t> output(n);

    input.reserve(n);

    // It looks like that currently tiny-dnn does not shuffle data
    // during training, so we provide a shuffled data set to tiny-dnn
    Rcpp::IntegerVector ind = Rcpp::sample(n, n, false, R_NilValue, false);

    // Copy data
    vec_t rowx(p);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < p; j++)
        {
            rowx[j] = x(ind[i], j);
        }
        input.push_back(rowx);

        output[i] = y[ind[i]];
    }

    std::shared_ptr<tiny_dnn::optimizer> opt_ptr = get_optimizer(opt);

    timer t;
    int epoch = 0;

    net->train<cross_entropy>(*opt_ptr, input, output, batch_size, epochs,
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
Rcpp::IntegerVector net_seq_classification_predict(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x
)
{
    using namespace tiny_dnn;

    const int n = x.nrow();
    const int p = x.ncol();

    Rcpp::IntegerVector pred(n);
    vec_t row(p);

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < p; j++)
        {
            row[j] = x(i, j);
        }
        pred[i] = net->predict_label(row);
    }

    return pred;
}
