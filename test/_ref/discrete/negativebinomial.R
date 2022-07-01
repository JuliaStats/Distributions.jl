
NegativeBinomial <- R6Class("NegativeBinomial",
    inherit = DiscreteDistribution,
    public = list(
        names = c("r", "p"),
        r = NA,
        p = NA,
        initialize = function(r=1, p=0.5) {
            self$r <- r
            self$p <- p
        },
        supp = function(){ c(0, Inf) },
        properties = function(){
            r <- self$r
            p <- self$p
            list(succprob=p,
                 failprob=1.0-p,
                 mean=(1-p) * r / p,
                 var=(1-p) * r / p^2,
                 skewness=(2-p) / sqrt((1-p) * r),
                 kurtosis=6 / r + p^2 / ((1-p)*r))
        },
        pdf = function(x, log=FALSE){ dnbinom(x, self$r, self$p, log=log)},
        cdf = function(x) { pnbinom(x, self$r, self$p) },
        quan = function(v) { qnbinom(v, self$r, self$p) }
    )
)
