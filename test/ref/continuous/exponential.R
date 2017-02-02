
Exponential <- R6Class("Exponential",
    inherit = ContinuousDistribution,
    public = list(
        names = c("theta"),
        theta = NA,
        beta = NA,
        initialize = function(s=1) {
            self$theta <- s
            self$beta <- 1 / s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            s <- self$theta
            list(scale=s,
                 rate=self$beta,
                 mean=s,
                 median=s * log(2),
                 var=s^2,
                 skewness=2.0,
                 kurtosis=6.0,
                 entropy=1.0 + log(s))
        },
        pdf = function(x, log=FALSE) { dexp(x, self$beta, log=log) },
        cdf = function(x) { pexp(x, self$beta) },
        quan = function(v) { qexp(v, self$beta) }
    )
)
