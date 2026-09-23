
SkewNormal <- R6Class("SkewNormal",
    inherit = ContinuousDistribution,
    public = list(
        names = c("xi", "omega", "alpha"),
        xi = NA,
        omega = NA,
        alpha = NA,
        initialize = function(xi=0, omega=1, alpha=0) {
            self$xi <- xi
            self$omega <- omega
            self$alpha <- alpha
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            # cumulants from sn (kappa_1..kappa_4), independent of the Julia formulas
            k <- sn:::sn.cumulants(dp=c(self$xi, self$omega, self$alpha), n=4)
            list(mean = k[1],
                 var = k[2],
                 std = sqrt(k[2]),
                 skewness = k[3] / k[2]^1.5,
                 kurtosis = k[4] / k[2]^2)
        },
        pdf = function(x, log=FALSE){ sn::dsn(x, xi=self$xi, omega=self$omega, alpha=self$alpha, log=log) },
        cdf = function(x){ sn::psn(x, xi=self$xi, omega=self$omega, alpha=self$alpha) },
        quan = function(v){ sn::qsn(v, xi=self$xi, omega=self$omega, alpha=self$alpha) }
    )
)
