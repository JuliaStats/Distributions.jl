
ZeroInflatedPoisson <- R6Class("ZeroInflatedPoisson",
    inherit = DiscreteDistribution,
    public = list(
        names = c("lambda", "p"),
        lambda = NA,
        p = NA,
        initialize = function(lambda = 1, p = 0) {
            self$lambda <- lambda
            self$p <- p
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            lam <- self$lambda
            p <- self$p
            list(rate = lam,
                 excessprob = p,
                 mean = (1 - p) * lam,
                 var = lam * (1 - p) * (1 + p * lam)
            )
        },
        pdf = function(x, log=FALSE) {
            VGAM::dzipois(x, self$lambda, pstr0 = self$p, log = log)
        },
        cdf = function(x) {
            VGAM::pzipois(x, self$lambda, pstr0 = self$p)
        },
        quan = function(v) {
            VGAM::qzipois(v, self$lambda, pstr0 = self$p)
        }
    )
)
