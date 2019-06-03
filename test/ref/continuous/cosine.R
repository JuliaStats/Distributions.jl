
Cosine <- R6Class("Cosine",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u=0, s=1) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(self$mu - self$sigma, self$mu + self$sigma) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            list(location = u,
                 scale = s,
                 mean = u,
                 median = u,
                 mode = u,
                 var = s^2 * (1/3 - 2/pi^2),
                 skewness = 0,
                 kurtosis = 1.2 * (90 - pi^4) / (pi^2 - 6)^2)
        },
        pdf = function(x, log=FALSE) {
            s <- self$sigma
            z <- (x - self$mu) / s
            p <- (1 + cospi(z)) / (2 * s)
            if (log) { base::log(p) } else { p }
        },
        cdf = function(x) {
            s <- self$sigma
            z <- (x - self$mu) / s
            (z+1)/2 + sinpi(z) / (2*pi)
        },
        quan = function(v) {
            invf <- function(u) {
                ret <- uniroot(
                    function(z) { (z+1)/2 + sinpi(z) / (2*pi) - u },
                    c(-1, 1), tol=1e-13)
                ret$root
            }
            r <- vapply(v, invf, 0.0)
            self$mu + r * self$sigma
        }
    )
)
