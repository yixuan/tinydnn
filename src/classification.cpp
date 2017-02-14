#include <tiny_dnn/tiny_dnn.h>
#include <Rcpp.h>
#include "layers.h"

// [[Rcpp::export]]
SEXP net_seq_classification_fit(
    Rcpp::XPtr< tiny_dnn::network<tiny_dnn::sequential> > net,
    Rcpp::NumericMatrix x,
    Rcpp::IntegerVector y,
    int batch_size,
    int epochs,
    std::string optimizer,
    bool verbose
)
{
    using namespace tiny_dnn;

    const int n = x.nrow();
    const int p = x.ncol();

    std::vector<vec_t> input;
    std::vector<label_t> output(n);

    input.reserve(n);

    // Copy data
    std::copy(y.begin(), y.end(), output.begin());

    vec_t rowx(p);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < p; j++)
        {
            rowx[j] = x(i, j);
        }
        input.push_back(rowx);
    }

    adagrad opt;

    timer t;
    int epoch = 0;

    net->train<cross_entropy>(opt, input, output, batch_size, epochs,
        // called for each mini-batch
        [&]() {

        },
        // called for each epoch
        [&]() {
            if(verbose)
            {
                Rcpp::Rcout << "[Epoch " << epoch << "]: " << t.elapsed() << "s" << std::endl;
                t.restart();
                epoch++;
            }
        });

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
