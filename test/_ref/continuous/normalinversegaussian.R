NormalInverseGaussian <- R6Class("NormalInverseGaussian",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "alpha", "beta", "delta"),
        mu = NA,
        alpha = NA,
        beta = NA,
        delta = NA,
        gamma = NA,
        initialize = function(u, a, b, d) {
            self$mu <- u
            self$alpha <- a
            self$beta <- b
            self$delta <- d
            self$gamma <- sqrt(a^2 - b^2)
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            a <- self$alpha
            b <- self$beta
            d <- self$delta
            g <- self$gamma
            list(mean = u + d * b / g,
                 var = d * a^2 / g^3,
                 skewness = 3 * b / a / sqrt(d * g),
                 kurtosis = 3 * (1 + 4 * b^2 / a^2) / (d * g))
        },
        pdf = function(x, log=FALSE) {
            fBasics::dnig(x, self$alpha, self$beta, self$delta, self$mu, log=log)
        },
        cdf = function(x) {
            fBasics::pnig(x, self$alpha, self$beta, self$delta, self$mu)
        },
        quan = function(v) {
            fBasics::qnig(v, self$alpha, self$beta, self$delta, self$mu)
        }
    )
)
