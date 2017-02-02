
NoncentralT <- R6Class("NoncentralT",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu", "ncp"),
        nu = NA,
        ncp = NA,
        initialize = function(k, ncp) {
            self$nu <- k
            self$ncp <- ncp
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() { list() },
        pdf = function(x, log=FALSE) { dt(x, self$nu, self$ncp, log=log) },
        cdf = function(x) { pt(x, self$nu, self$ncp) },
        quan = function(v) { qt(v, self$nu, self$ncp) }
    )
)
