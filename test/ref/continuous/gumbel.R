
Gumbel <- R6Class("Gumbel",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "beta"),
        mu = NA,
        beta = NA,
        initialize = function(u=0, b=1) {
            self$mu <- u
            self$beta <- b
        },
        supp = function(){ c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            b <- self$beta
            g <- 0.57721566490153286
            list(location = u,
                 scale = b,
                 mean = u + b * g,
                 median = u - b * log(log(2)),
                 mode = u,
                 var = pi^2 * b^2 / 6,
                 skewness = 1.13954709940464866,
                 kurtosis = 2.4)
        },
        pdf = function(x, log=FALSE){ dgumbel(x, self$mu, self$beta, log=log) },
        cdf = function(x){ pgumbel(x, self$mu, self$beta) },
        quan = function(v){ qgumbel(v, self$mu, self$beta) }
    )
)
