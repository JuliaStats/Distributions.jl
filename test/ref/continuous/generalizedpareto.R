
GeneralizedPareto <- R6Class("GeneralizedPareto",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma", "xi"),
        mu = NA,
        sigma = NA,
        xi = NA,
        initialize = function(a1=NA, a2=NA, a3=NA) {
            if (is.na(a1)) {
                u <- 0; s <- 1; k <- 1
            } else if (is.na(a2)){
                u <- 0; s <- 1; k <- a1
            } else if (is.na(a3)) {
                u <- 0; s <- a1; k <- a2
            } else {
                u <- a1; s <- a2; k <- a3
            }
            self$mu <- u
            self$sigma <- s
            self$xi <- k
        },
        supp = function() {
            u <- self$mu
            s <- self$sigma
            k <- self$xi
            if (k >= 0) { c(u, Inf) } else { c(u, u - s/k) }
        },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            k <- self$xi
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
