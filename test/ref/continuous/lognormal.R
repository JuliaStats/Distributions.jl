
LogNormal <- R6Class("LogNormal",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u=0, s=1) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            es2 <- exp(s^2)
            list(meanlogx = u,
                 varlogx = s^2,
                 stdlogx = s,
                 mean = exp(u + s^2/2),
                 median = exp(u),
                 mode = exp(u - s^2),
                 var = (es2 - 1) * exp(2 * u + s^2),
                 skewness = (es2 + 2) * sqrt(es2 - 1),
                 kurtosis = es2^4 + 2 * es2^3 + 3 * es2^2 - 6,
                 entropy = (u + 1/2) + log(sqrt(2 * pi) * s))
        },
        pdf = function(x, log=FALSE){ dlnorm(x, self$mu, self$sigma, log=log) },
        cdf = function(x){ plnorm(x, self$mu, self$sigma) },
        quan = function(v){ qlnorm(v, self$mu, self$sigma) }
    )
)
