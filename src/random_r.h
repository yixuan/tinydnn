// Modified version of <tiny_dnn/util/random.h>, using R's RNG
#pragma once

#include <Rcpp.h>

namespace tiny_dnn {

template <typename Container>
inline int uniform_idx(const Container &t) {
  return int(R::unif_rand() * t.size());
}

inline bool bernoulli(float_t p) {
  return float_t(R::unif_rand()) <= p;
}

template <typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
  for (Iter it = begin; it != end; ++it) *it = R::runif(min, max);
}

template <typename Iter>
void gaussian_rand(Iter begin, Iter end, float_t mean, float_t sigma) {
  for (Iter it = begin; it != end; ++it) *it = R::rnorm(mean, sigma);
}

}  // namespace tiny_dnn
