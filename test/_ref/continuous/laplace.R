
Laplace <- R6Class("Laplace",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "beta"),
        mu = NA,
        beta = NA,
        initialize = function(u=0, b=1) {
            self$mu <- u
            self$beta <- b
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            b <- self$beta
            list(location = u,
                 scale = b,
                 mean = u,
                 median = u,
                 mode = u,
                 var = 2 * b^2,
                 skewness = 0,
                 kurtosis = 3,
                 entropy = 1 + log(2 * b))
        },
        pdf = function(x, log=FALSE){ dlaplace(x, self$mu, self$beta, log=log) },
        cdf = function(x){ plaplace(x, self$mu, self$beta) },
        quan = function(v){ qlaplace(v, self$mu, self$beta) }
    )
)
