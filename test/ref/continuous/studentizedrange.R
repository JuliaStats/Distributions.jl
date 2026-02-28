
StudentizedRange <- R6Class("StudentizedRange",
    inherit = ContinuousDistribution,
    public = list(
        names = c("nu", "k"),
        nu = NA,
        k = NA,
        initialize = function(nu, k) {
            self$nu <- nu
            self$k <- k
        },
        supp = function() { c(0, Inf) },
        properties = function() {
            d1 <- self$nu
            k <- self$k
        },
        pdf = function(x, log=FALSE) {
            if (log) {
                ptukey(x, self$k, self$nu, log=TRUE) + log(pracma::fderiv(ptukey, x, n=1, nmeans=self$k, df=self$nu, log=TRUE))
            } else {
                pracma::fderiv(ptukey, x, n=1, nmeans=self$k, df=self$nu, log=FALSE)
            }
        },
        cdf = function(x) { ptukey(x, self$k, self$nu) },
        quan = function(v) { qtukey(v, self$k, self$nu) }
    )
)
