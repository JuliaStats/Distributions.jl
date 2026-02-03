LogLogistic <- R6Class("LogLogistic",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a, b) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function() { c(0, Inf) },
        properties = function() { },
        pdf = function(x, log=FALSE) { VGAM::dfisk(x, scale=self$alpha, shape=self$beta, log=log) },
        cdf = function(x) { VGAM::pfisk(x, scale=self$alpha, shape=self$beta) },
        quan = function(v) { VGAM::qfisk(v, scale=self$alpha, shape=self$beta) }
    )
)
