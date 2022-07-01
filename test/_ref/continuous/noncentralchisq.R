
NoncentralChisq <- R6Class("NoncentralChisq",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu", "ncp"),
        nu = NA,
        ncp = NA,
        initialize = function(k, ncp) {
            self$nu <- k
            self$ncp <- ncp
        },
        supp = function() { c(0, Inf) },
        properties = function() { list() },
        pdf = function(x, log=FALSE) { dchisq(x, self$nu, self$ncp, log=log) },
        cdf = function(x) { pchisq(x, self$nu, self$ncp) },
        quan = function(v) { qchisq(v, self$nu, self$ncp) }
    )
)
