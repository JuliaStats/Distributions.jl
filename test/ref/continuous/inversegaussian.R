
InverseGaussian = R6Class("InverseGaussian",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "lambda"),
        mu = NA,
        lambda = NA,
        initialize = function(u=1, lambda=1) {
            self$mu <- u
            self$lambda <- lambda
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            u <- self$mu
            l <- self$lambda
            list(shape = l,
                 mean = u,
                 mode = u * ( (1 + 9 * u^2 / (4 * l^2))^0.5 - (3 * u) / (2 * l) ),
                 var = u^3 / l,
                 skewness = 3 * sqrt(u / l),
                 kurtosis = 15 * u / l)
        },
        pdf = function(x, log=FALSE){ statmod::dinvgauss(x, self$mu, self$lambda, log=log) },
        cdf = function(x){ statmod::pinvgauss(x, self$mu, self$lambda) },
        quan = function(v){ statmod::qinvgauss(v, self$mu, self$lambda) }
    )
)
