
StudentizedRange <- R6Class("StudentizedRange",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu", "k"),
        nu = NA,
        k = NA,
        initialize = function(nu, k) {
            self$nu <- nu
            self$k <- k
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            d1 <- self$nu
            k <- self$k
        },
        pdf = function(x, log=FALSE) rep(NaN, length(x)),
        cdf = function(x) { ptukey(x, self$k, self$nu) },
        quan = function(v) { qtukey(v, self$k, self$nu) }
    )
)
