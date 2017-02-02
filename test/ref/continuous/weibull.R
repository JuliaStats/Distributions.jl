
Weibull <- R6Class("Weibull",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "theta"),
        alpha = NA,
        theta = NA,
        initialize = function(a=1, s=1) {
            self$alpha <- a
            self$theta <- s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            a <- self$alpha
            s <- self$theta
            var.val <- s^2 * (gamma(1 + 2 / a) - gamma(1 + 1 / a)^2)
            gv <- -digamma(1)
            list(shape = a,
                 scale = s,
                 mean = s * gamma(1.0 + 1 / a),
                 median = s * (log(2) ^ (1 / a)),
                 var = var.val,
                 entropy = gv * (1 - 1 / a) + log(s / a) + 1)
        },
        pdf = function(x, log=FALSE) { dweibull(x, self$alpha, self$theta, log=log) },
        cdf = function(x) { pweibull(x, self$alpha, self$theta) },
        quan = function(v) { qweibull(v, self$alpha, self$theta) }
    )
)
