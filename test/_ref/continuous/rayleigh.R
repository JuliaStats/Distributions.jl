
Rayleigh <- R6Class("Rayleigh",
    inherit = ContinuousDistribution,
    public = list(
        names = c("sigma"),
        sigma = NA,
        initialize = function(s=1) {
            self$sigma <- s
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            s <- self$sigma
            list(scale = s,
                 mean = s * sqrt(pi / 2),
                 median = s * sqrt(2 * log(2)),
                 mode = s,
                 var = s^2 * (4 - pi) / 2,
                 skewness = 0.631110657818937,
                 kurtosis = 0.245089300687638,
                 entropy = 0.94203424217079 + log(s))
        },
        pdf = function(x, log=FALSE) { drayleigh(x, self$sigma, log=log) },
        cdf = function(x){ prayleigh(x, self$sigma) },
        quan = function(v){ qrayleigh(v, self$sigma) }
    )
)
