
Logistic <- R6Class("Logistic",
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
            u <- self$mu
            s <- self$sigma
            list(location = u,
                 scale = s,
                 mean = u,
                 median = u,
                 mode = u,
                 var = s^2 * pi^2 / 3,
                 skewness = 0,
                 kurtosis = 1.2,
                 entropy = log(s) + 2)
        },
        pdf = function(x, log=FALSE){ dlogis(x, self$mu, self$sigma, log=log) },
        cdf = function(x){ plogis(x, self$mu, self$sigma) },
        quan = function(v){ qlogis(v, self$mu, self$sigma) }
    )
)
