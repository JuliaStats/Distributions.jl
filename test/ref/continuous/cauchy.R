
Cauchy <- R6Class("Cauchy",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u, s) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            list(location=self$mu,
                 scale=self$sigma,
                 median=self$mu,
                 entropy=log(self$sigma) + log(4 * pi))
        },
        pdf = function(x, log=FALSE) { dcauchy(x, self$mu, self$sigma, log=log) },
        cdf = function(x) { pcauchy(x, self$mu, self$sigma) },
        quan = function(v) { qcauchy(v, self$mu, self$sigma) }
    )
)
