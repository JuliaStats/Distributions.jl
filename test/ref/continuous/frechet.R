
Frechet <- R6Class("Frechet",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a=1, b=1) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            a <- self$alpha
            b <- self$beta
            g1 <- ifelse(a > 1, gamma(1 - 1/a), NaN)
            g2 <- ifelse(a > 2, gamma(1 - 2/a), NaN)
            g3 <- ifelse(a > 3, gamma(1 - 3/a), NaN)
            g4 <- ifelse(a > 4, gamma(1 - 4/a), NaN)
            gam <- 0.57721566490153286
            list(shape = a,
                 scale = b,
                 mean = ifelse(a > 1, b * g1, Inf),
                 median = b / log(2)^(1/a),
                 mode = b * (a / (1 + a))^(1/a),
                 var = ifelse(a > 2, b^2 * (g2 - g1^2), Inf),
                 skewness = if (a > 3) {
                     (g3 - 3 * g2 * g1 + 2 * g1^3) / (g2 - g1^2)^1.5
                 } else { Inf },
                 kurtosis = if (a > 4) {
                     (g4 - 4 * g3 * g1 + 3 * g2^2) / (g2 - g1^2)^2 - 6
                 } else { Inf },
                 entropy = 1 + gam / a + gam + log(b / a))
        },
        pdf = function(x, log=FALSE) { VGAM::dfrechet(x, shape=self$alpha, scale=self$beta, log=log) },
        cdf = function(x) { VGAM::pfrechet(x, shape=self$alpha, scale=self$beta) },
        quan = function(v) { VGAM::qfrechet(v, shape=self$alpha, scale=self$beta) }
    )
)
