
Chi <- R6Class("Chi",
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
            u <- sqrt(2) * gamma((k+1)/2) / gamma(k/2)
            v <- k - u^2
            m3 <- u / v^1.5 * (1 - 2 * v)
            m4 <- (2 / v) * (1 - u * sqrt(v) * m3 - v)
            list(dof = k,
                 mode = ifelse(k >= 1, sqrt(k - 1), NaN),
                 mean = u,
                 var = v,
                 skewness = m3,
                 kurtosis = m4,
                 entropy = lgamma(k/2) + (k - log(2) - (k-1)*psigamma(k/2, 0)) / 2
                )
        },
        pdf = function(x, log=FALSE) { chi::dchi(x, self$nu, log=log) },
        cdf = function(x) { chi::pchi(x, self$nu) },
        quan = function(v) { chi::qchi(v, self$nu) }
    )
)
