
Skellam <- R6Class("Skellam",
    inherit = DiscreteDistribution,
    public = list(
        names = c("mu1", "mu2"),
        mu1 = NA,
        mu2 = NA,
        initialize = function(mu1=1, mu2=mu1) {
            self$mu1 <- mu1
            self$mu2 <- mu2
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u1 <- self$mu1
            u2 <- self$mu2
            list(mean=u1 - u2,
                 var=u1 + u2,
                 skewness=(u1 - u2) / (u1 + u2)^1.5,
                 kurtosis=1/(u1 + u2))
        },
        pdf = function(x, log=FALSE) { skellam::dskellam(x, self$mu1, self$mu2, log=log) },
        cdf = function(x){ skellam::pskellam(x, self$mu1, self$mu2) },
        quan = function(v){ skellam::qskellam(v, self$mu1, self$mu2) }
    )
)
