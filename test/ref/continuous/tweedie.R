# Tweedie Distribution
# Using R's tweedie package for reference implementation

Tweedie <- R6Class("Tweedie",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma", "p"),
        mu = NA,
        sigma = NA,
        p = NA,
        initialize = function(mu, sigma, p) {
            self$mu <- mu
            self$sigma <- sigma
            self$p <- p
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            mu <- self$mu
            sigma <- self$sigma
            p <- self$p
            list(location=mu,
                 scale=sigma,
                 shape=p,
                 mean=mu,
                 var=mu^p * sigma^2,
                 skewness=p * sigma / sqrt(mu ^ (2 - p)),
                 kurtosis=p * (2 * p - 1) * sigma^2 / mu ^ (2 - p))
        },
        pdf = function(x, log=FALSE) {
            val <- tweedie::dtweedie(x, mu=self$mu, phi=self$sigma^2, power=self$p)
            if (log) {
                return(log(val))
            } else {
                return(val)
            }
        },
        cdf = function(x) {
            tweedie::ptweedie(x, mu=self$mu, phi=self$sigma^2, power=self$p)
        },
        quan = function(v) {
            tweedie::qtweedie(v, mu=self$mu, phi=self$sigma^2, power=self$p)
        }
    )
)
