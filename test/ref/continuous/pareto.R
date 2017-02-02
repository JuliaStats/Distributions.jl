
Pareto <- R6Class("Pareto",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a=1, b=1) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function() { c(self$beta, Inf) },
        properties = function() {
            a <- self$alpha
            b <- self$beta
            list(shape = a,
                 scale = b,
                 mean = if (a > 1) { a * b / (a - 1) } else { Inf },
                 median = b * 2^(1/a),
                 mode = b,
                 var = if (a > 2) {
                     a * b^2 / ((a - 1)^2 * (a - 2))
                 } else { Inf },
                 skewness = if (a > 3) {
                     2 * (1 + a) / (a - 3) * sqrt((a - 2) / a)
                 } else { NaN },
                 kurtosis = if (a > 4) {
                     6 * (a^3 + a^2 - 6*a - 2) /
                     (a * (a - 3) * (a - 4))
                 } else { NaN },
                 entropy = (1 + 1 / a) + log(b / a))
        },
        pdf = function(x, log=FALSE){ dpareto(x, self$alpha, self$beta, log=log) },
        cdf = function(x){ ppareto(x, self$alpha, self$beta) },
        quan = function(v){ qpareto(v, self$alpha, self$beta) }
    )
)
