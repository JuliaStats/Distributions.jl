library("ExtDist")

JohnsonSU <- R6Class("JohnsonSU",
    inherit = ContinuousDistribution,
    public = list(names = c("xi", "lambda", "gamma", "delta"),
        xi = NA,
        lambda = NA,
        gamma = NA,
        delta = NA,
        initialize = function(xi = 0, lambda = 1, gamma = 0, delta = 1) {
            self$xi <- xi
            self$lambda <- lambda
            self$gamma <- gamma
            self$delta <- delta
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() { list() },
        pdf = function(x, log = FALSE) {
            p <- dJohnsonSU(x, xi = self$xi, lambda = self$lambda, gamma = self$gamma, delta = self$delta)
            if (log) {
                result <- log(p)
            } else {
                result <- p
            }
            return(result)
        },
        cdf = function(x) {
            pJohnsonSU(x, xi = self$xi, lambda = self$lambda, gamma = self$gamma, delta = self$delta)
        },
        quan = function(x) {
            qJohnsonSU(x, xi = self$xi, lambda = self$lambda, gamma = self$gamma, delta = self$delta)
        }
    )
)
