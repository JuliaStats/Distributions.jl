set.seed(8675309)
S   <- matrix(c(1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0), 3, 3)
X   <- StanHeaders::stanFunction("inv_wishart_rng", nu=4, S=S)
invwishart_output <- lapply(
    c(4.5, 8.2, 10.0),
    function(nu, X, S) {
        cat("InverseWishart(", nu, ", S)\n", sep="")
        lpdf <- StanHeaders::stanFunction("inv_wishart_lpdf", W=X, nu=nu, S=S)
        list("dims"   = dim(S),
             "params" = list(nu, c(S)),
             "X"      = c(X),
             "lpdf"   = lpdf)
    },
    X,
    S
)
jsonlite::write_json(invwishart_output, "jsonfiles/InverseWishart_stan_output.json", digits = 20, pretty = TRUE)

set.seed(8675309)
X      <- StanHeaders::stanFunction("lkj_corr_rng", K=3, eta=1)
lkj_output <- lapply(
    c(0.5, 1.0, 3.4),
    function(eta, X) {
        cat("LKJ(", nrow(X), ", ", eta, ")\n", sep="")
        lpdf <- StanHeaders::stanFunction("lkj_corr_lpdf", y=X, eta=eta)
        list("dims"   = dim(X),
             "params" = list(nrow(X), eta),
             "X"      = c(X),
             "lpdf"   = lpdf)
    },
    X
)
jsonlite::write_json(lkj_output, "jsonfiles/LKJ_stan_output.json", digits = 20, pretty = TRUE)

set.seed(8675309)
S   <- matrix(c(1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 1.0), 3, 3)
X   <- StanHeaders::stanFunction("wishart_rng", nu=4, S=S)
wishart_output <- lapply(
    c(4.5, 8.2, 10.0),
    function(nu, X, S) {
        cat("Wishart(", nu, ", S)\n", sep="")
        lpdf <- StanHeaders::stanFunction("wishart_lpdf", W=X, nu=nu, S=S)
        list("dims"   = dim(S),
             "params" = list(nu, c(S)),
             "X"      = c(X),
             "lpdf"   = lpdf)
    },
    X,
    S
)
jsonlite::write_json(wishart_output, "jsonfiles/Wishart_stan_output.json", digits = 20, pretty = TRUE)
