
Gamma <- R6Class("Gamma",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "theta"),
        alpha = NA,
        theta = NA,
        beta = NA,
        initialize = function(a=1, s=1) {
            self$alpha <- a
            self$theta <- s
            self$beta <- 1 / s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            a <- self$alpha
            s <- self$theta
            list(shape=a,
                 scale=s,
                 rate=1/s,
                 mean=a * s,
                 var=a * s^2,
                 skewness=2 / sqrt(a),
                 kurtosis=6 / a,
                 entropy=a + log(s) + lgamma(a) + (1 - a) * digamma(a)
            )
        },
        pdf = function(x, log=FALSE) { dgamma(x, self$alpha, self$beta, log=log) },
        cdf = function(x) { pgamma(x, self$alpha, self$beta) },
        quan = function(v) { qgamma(v, self$alpha, self$beta) }
    )
)

Erlang = list(new = Gamma$new)
