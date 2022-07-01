
InverseGamma <- R6Class("InverseGamma",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a=1, b=1) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function(){ c(0, Inf) },
        properties = function() {
            a <- self$alpha
            b <- self$beta
            list(shape = a,
                 scale = b,
                 rate = 1 / b,
                 mean = if (a > 1) { b / (a - 1) } else { Inf },
                 mode = b / (a + 1),
                 var = if (a > 2) {
                     b^2 / ((a - 1)^2 * (a - 2))
                 } else { Inf },
                 skewness = if (a > 3) {
                     4 * sqrt(a - 2) / (a - 3)
                 } else { NaN },
                 kurtosis = if (a > 4) {
                     (30 * a - 66) / ((a - 3) * (a - 4))
                 } else { NaN },
                 entropy = a + log(b) + lgamma(a) - (1 + a) * digamma(a))
        },
        pdf = function(x, log=FALSE){ dinvgamma(x, self$alpha, self$beta, log=log) },
        cdf = function(x){ pinvgamma(x, self$alpha, self$beta) },
        quan = function(v){ qinvgamma(v, self$alpha, self$beta) }
    )
)
