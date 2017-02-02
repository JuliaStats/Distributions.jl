
Poisson <- R6Class("Poisson",
    inherit = DiscreteDistribution,
    public = list(
        names = c("lambda"),
        lambda = NA,
        initialize = function(lambda=1) {
            self$lambda <- lambda
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            lam <- self$lambda
            list(rate=lam,
                 mean=lam,
                 var=lam,
                 skewness=1/sqrt(lam),
                 kurtosis=1/lam)
        },
        pdf = function(x, log=FALSE) { dpois(x, self$lambda, log=log) },
        cdf = function(x){ ppois(x, self$lambda) },
        quan = function(v){ qpois(v, self$lambda) }
    )
)
