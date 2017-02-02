
Levy <- R6Class("Levy",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u=0, s=1) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(self$mu, Inf) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            erfcinv <- function (x) qnorm(x/2, lower = FALSE)/sqrt(2)
            list(location = u,
                 mode = u + s / 3,
                 mean = Inf,
                 var = Inf,
                 skewness = NaN,
                 kurtosis = NaN,
                 # 0.47693627620447 = erfc^{-1}(0.5)
                 median = u + (s/2) / (0.47693627620447)^2)
        },
        pdf = function(x, log=FALSE){ VGAM::dlevy(x, self$mu, self$sigma, log.arg=log) },
        cdf = function(x) { VGAM::plevy(x, self$mu, self$sigma) },
        quan = function(v) { VGAM::qlevy(v, self$mu, self$sigma) }
    )
)
