
TDist <- R6Class("TDist",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu"),
        nu = NA,
        initialize = function(nu) {
            self$nu <- nu
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            nu <- self$nu
            a <- (nu + 1) / 2
            ent.val <- a * (digamma(a) - digamma(nu / 2)) +
                       log(nu) / 2 + lbeta(nu / 2, 1 / 2)
            list(dof=nu,
                 mean=ifelse(nu > 1, 0, NaN),
                 median=0,
                 var=ifelse(nu > 2, nu / (nu - 2),
                     ifelse(nu > 1, Inf, NaN)),
                 skewness = ifelse(nu > 3, 0, NaN),
                 kurtosis = ifelse(nu > 4, 6 / (nu - 4),
                            ifelse(nu > 2, Inf, NaN)),
                 entropy = ent.val)
        },
        pdf = function(x, log=FALSE) { dt(x, self$nu, log=log) },
        cdf = function(x) { pt(x, self$nu) },
        quan = function(v) { qt(v, self$nu) }
    )
)
