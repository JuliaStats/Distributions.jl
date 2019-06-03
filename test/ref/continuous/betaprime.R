
BetaPrime <- R6Class("BetaPrime",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a=1, b=a) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            a <- self$alpha
            b <- self$beta
            list(mean = if (b > 1) { a / (b - 1) } else { NaN },
                 mode = if (a > 1) {
                     (a - 1) / (b + 1)
                 } else { 0 },
                 var = if (b > 2) {
                     a * (a + b - 1) / ((b - 2) * (b - 1)^2)
                 } else { NaN },
                 skewness = if (b > 3) {
                     2 * (2 * a + b - 1) / (b - 3) *
                     sqrt((b - 2) / (a * (a + b - 1)))
                 } else { NaN })
        },
        pdf = function(x, log=FALSE){ dbetapr(x, self$alpha, self$beta, log=log) },
        cdf = function(x) { pbetapr(x, self$alpha, self$beta) },
        quan = function(v) { qbetapr(v, self$alpha, self$beta) }
    )
)
