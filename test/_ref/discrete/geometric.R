
Geometric <- R6Class("Geometric",
    inherit = DiscreteDistribution,
    public = list(
        names = c("p"),
        p = NA,
        initialize = function(p=0.5) {
            self$p <- p
        },
        supp = function(){ c(0, Inf) },
        properties = function() {
            p <- self$p
            list(succprob=p,
                 failprob=1.0-p,
                 mean=(1-p)/p,
                 var=(1-p)/p^2,
                 skewness=(2-p)/sqrt(1-p),
                 kurtosis=6 + p^2/(1-p),
                 entropy=-((1-p)*log(1-p) + p * log(p))/p)
        },
        pdf = function(x, log=FALSE) { dgeom(x, self$p, log=log) },
        cdf = function(x) { pgeom(x, self$p) },
        quan = function(v){ qgeom(v, self$p) }
    )
)
