library(rstan)
library(jsonlite)

invwishart <-
  '
  functions {
  real inv_wish_lpdf(matrix W, real nu, matrix Sigma) {
    return inv_wishart_lpdf(W | nu, Sigma);
  }
  matrix inv_wish_rng(real nu, matrix Sigma) {
    return inv_wishart_rng(nu, Sigma);
  }
}
'
expose_stan_functions(stanc(model_code = invwishart))

set.seed(8675309)
S   <- matrix(c(1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0), 3, 3)
d   <- nrow(S)
X   <- inv_wish_rng(4, S)
dfs <- c(4.5, 8.2, 10.0)
K   <- length(dfs)
invwishart_output <- vector(mode = "list", length = K)
for (i in 1:K){
    lpdf <- inv_wish_lpdf(X, dfs[i], S)
    invwishart_output[[i]] <- list("dims"   = c(d, d),
                                   "params" = list(dfs[i], c(S)),
                                   "X"      = c(X),
                                   "lpdf"   = lpdf)
}
write_json(invwishart_output, "jsonfiles/InverseWishart_stan_output.json", digits = 20, pretty = TRUE)
