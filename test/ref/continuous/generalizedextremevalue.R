
GeneralizedExtremeValue <- R6Class("GeneralizedExtremeValue",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma", "xi"),
        mu = NA,
        sigma = NA,
        xi = NA,
        initialize = function(u, s, k) {
            self$mu <- u
            self$sigma <- s
            self$xi <- k
        },
        supp = function() {
            k <- self$xi
            if (k > 0) {
                c(self$mu - self$sigma / k, Inf)
            } else if (k == 0) {
                c(-Inf, Inf)
            } else if (k < 0) {
                c(-Inf, self$mu - self$sigma / k)
            }
        },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            k <- self$xi
            gam <- 0.57721566490153286
            zeta3 <- 1.202056903159594
            g1 <- if (k < 1) gamma(1 - k)
            g2 <- if (2*k < 1) gamma(1 - 2*k)
            g3 <- if (3*k < 1) gamma(1 - 3*k)
            g4 <- if (4*k < 1) gamma(1 - 4*k)
            ent <- log(s) + gam * k + (gam + 1)
            if (k == 0) {
                list(location = u,
                     scale = s,
                     shape = k,
                     mean = u + s * gam,
                     median = u - s * log(log(2)),
                     mode = u,
                     var = (s * pi)^2 / 6,
                     skewness = 1.1395470994046487,
                     kurtosis = 12/5,
                     entropy = ent)
            } else {
                list(location = u,
                     scale = s,
                     shape = k,
                     mean = if (k < 1) { u + s * (g1 - 1) / k } else { Inf },
                     median = u + s * (log(2)^(-k) - 1) / k,
                     mode = u + s * ((1 + k)^(-k) - 1) / k,
                     var = if (2*k < 1) { s^2 * (g2 - g1^2) / k^2 } else { Inf },
                     skewness = if (3*k < 1) {
                         sign(k) * (g3 - 3*g1*g2 + 2*g1^3) / (g2-g1^2)^1.5
                     } else { Inf },
                     kurtosis = if (4*k < 1) {
                         (g4 - 4*g1*g3 + 6*g2*g1^2 - 3*g1^4) / (g2-g1^2)^2 - 3
                     } else { Inf },
                     entropy = ent)
            }
        },
        pdf = function(x, log=FALSE) { dgev(x, self$mu, self$sigma, self$xi, log=log) },
        cdf = function(x) { pgev(x, self$mu, self$sigma, self$xi) },
        quan = function(v) { qgev(v, self$mu, self$sigma, self$xi) }
    )
)
