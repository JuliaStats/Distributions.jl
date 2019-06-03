
NoncentralF <- R6Class("NoncentralF",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu1", "nu2", "ncp"),
        nu1 = NA,
        nu2 = NA,
        ncp = NA,
        initialize = function(k1, k2, ncp) {
            self$nu1 <- k1
            self$nu2 <- k2
            self$ncp <- ncp
        },
        supp = function() { c(0, Inf) },
        properties = function() { list() },
        pdf = function(x, log=FALSE) { df(x, self$nu1, self$nu2, self$ncp, log=log) },
        cdf = function(x) { pf(x, self$nu1, self$nu2, self$ncp) },
        quan = function(v) { qf(v, self$nu1, self$nu2, self$ncp) }
    )
)
