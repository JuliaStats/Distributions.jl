
NoncentralBeta <- R6Class("NoncentralBeta",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta", "ncp"),
        alpha = NA,
        beta = NA,
        ncp = NA,
        initialize = function(a, b, ncp) {
            self$alpha <- a
            self$beta <- b
            self$ncp <- ncp
        },
        supp = function() { c(0, 1) },
        properties = function() { list() },
        pdf = function(x, log=FALSE) { dbeta(x, self$alpha, self$beta, self$ncp, log=log) },
        cdf = function(x) { pbeta(x, self$alpha, self$beta, self$ncp) },
        quan = function(v) { qbeta(v, self$alpha, self$beta, self$ncp) }
    )
)
