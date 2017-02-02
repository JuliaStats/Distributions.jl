
GeneralizedPareto <- R6Class("GeneralizedPareto",
    inherit = ContinuousDistribution,
    public = list(
        names = c("xi", "sigma", "mu"),
        xi = NA,
        sigma = NA,
        mu = NA,
        initialize = function(m, s, k) {
            self$xi <- k
            self$sigma <- s
            self$mu <- m
        },
        supp = function() {
            k <- self$xi
            s <- self$sigma
            u <- self$mu
            if (k >= 0) { c(u, Inf) } else { c(u, u - s/k) }
        },
        properties = function() {
            k <- self$xi
            s <- self$sigma
            u <- self$mu
            list(location = u,
                 scale = s,
                 shape = k,
                 mean = ifelse(k < 1, u + s / (1 - k), Inf),
                 median = u + s * (2^k - 1) / k,
                 var = if (2*k < 1) {
                     s^2 / ((1 - k)^2 * (1 - 2*k))
                 } else { Inf },
                 skewness = if (3*k < 1) {
                     2 * (1 + k) * sqrt(1 - 2*k) / (1 - 3*k)
                 } else { Inf },
                 kurtosis = if (4*k < 1) {
                     3 * (1 - 2*k) * (2*k^2 + k + 3) / (1 - 3*k) / (1 - 4*k) - 3
                 } else { Inf }
            )
        },
        pdf = function(x, log=FALSE) { dgpd(x, self$mu, self$sigma, self$xi, log=log) },
        cdf = function(x) { pgpd(x, self$mu, self$sigma, self$xi) },
        quan = function(v) { qgpd(v, self$mu, self$sigma, self$xi) }
    )
)
