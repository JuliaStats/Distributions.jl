
Rician <- R6Class("Rician",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu", "sigma"),
        nu = NA,
        sigma = NA,
        initialize = function(n=0, s=1) {
            self$nu <- n
            self$sigma <- s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            n <- self$nu
            s <- self$sigma
            list(scale = n^2 + 2 * s^2,
                 shape = n^2 / (2 * s^2))
        },
        pdf = function(x, log=FALSE) { VGAM::drice(x, self$sigma, self$nu, log=log) },
        cdf = function(x){ VGAM::price(x, self$sigma, self$nu) },
        quan = function(v){ VGAM::qrice(v, self$sigma, self$nu) }
    )
)
