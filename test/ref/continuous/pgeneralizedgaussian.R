PGeneralizedGaussian <- R6Class("PGeneralizedGaussian",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "alpha", "beta"),
        mu = NA,
        alpha = NA,
        beta = NA,
        initialize = function(m=0, a=sqrt(2), p=2) {
            self$mu <- m
            self$alpha <- a
            self$beta <- p
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            m <- self$mu
            a <- self$alpha
            p <- self$beta
            list(location=m,
                 scale=a,
                 shape=p,
                 mean=m,
                 var=a^2 * exp(lgamma(3/p) - lgamma(1/p)),
                 median=m,
                 mode=m,
                 entropy=1/p - log(p/(2 * a)) + lgamma(1/p),
                 skewness=0,
                 kurtosis=exp(lgamma(5/p) + lgamma(1/p) - 2 * lgamma(3/p)) - 3)
        },
        pdf = function(x, log=FALSE){ gnorm::dgnorm(x, self$mu, self$alpha, self$beta, log=log) },
        cdf = function(x){ gnorm::pgnorm(x, self$mu, self$alpha, self$beta) },
        quan = function(v) { gnorm::qgnorm(v, self$mu, self$alpha, self$beta) }
    )
)
