
FDist <- R6Class("FDist",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu1", "nu2"),
        nu1 = NA,
        nu2 = NA,
        initialize = function(nu1, nu2) {
            self$nu1 <- nu1
            self$nu2 <- nu2
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            d1 <- self$nu1
            d2 <- self$nu2
            list(mean = if (d2 > 2) { d2 / (d2 - 2) } else { NaN },
                 mode = if (d1 > 2) {
                     (d1 - 2) * d2 / (d1 * (d2 + 2))
                 } else { 0 },
                 var = if (d2 > 4) {
                     2 * d2^2 * (d1 + d2 - 2) /
                     ( d1 * (d2 - 2)^2 * (d2 - 4) )
                 } else { NaN },
                 skewness = if (d2 > 6) {
                     (2 * d1 + d2 - 2) * sqrt(8 * (d2 - 4)) /
                     ( (d2 - 6) * sqrt(d1 * (d1 + d2 - 2)) )
                 } else { NaN })
        },
        pdf = function(x, log=FALSE) { df(x, self$nu1, self$nu2, log=log) },
        cdf = function(x) { pf(x, self$nu1, self$nu2) },
        quan = function(v) { qf(v, self$nu1, self$nu2) }
    )
)
