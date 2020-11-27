library(rstan)
library(jsonlite)

lkj <-
  '
  functions {
  real lkj_cor_lpdf(matrix y, real eta) {
    return lkj_corr_lpdf(y | eta);
  }
  matrix lkj_cor_rng(int K, real eta) {
    return lkj_corr_rng(K, eta);
  }
}
'
expose_stan_functions(stanc(model_code = lkj))

set.seed(8675309)
d      <- 3
X      <- lkj_cor_rng(d, 1)
shapes <- c(0.5, 1.0, 3.4)
K      <- length(shapes)
lkj_output <- vector(mode = "list", length = K)
for (i in 1:K){
    lpdf <- lkj_cor_lpdf(X, shapes[i])
    lkj_output[[i]] <- list("dims"   = c(d, d),
                            "params" = list(d, shapes[i]),
                            "X"      = c(X),
                            "lpdf"   = lpdf)
}
write_json(lkj_output, "jsonfiles/LKJ_stan_output.json", digits = 20, pretty = TRUE)
