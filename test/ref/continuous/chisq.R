
Chisq <- R6Class("Chisq",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu"),
        nu = NA,
        initialize = function(nu) {
            self$nu <- nu
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            k <- self$nu
            list(dof=k,
                 mean=k,
                 var=2 * k,
                 skewness=sqrt(8 / k),
                 kurtosis=12 / k,
                 entropy=k / 2 + log(2) + lgamma(k/2) + (1 - k/2) * digamma(k/2))
        },
        pdf = function(x, log=FALSE) { dchisq(x, self$nu, log=log) },
        cdf = function(x) { pchisq(x, self$nu) },
        quan = function(v) { qchisq(v, self$nu) }
    )
)
