
Bernoulli <- R6Class("Bernoulli",
    inherit = DiscreteDistribution,
    public = list(
        names = c("p"),
        p = NA,
        initialize = function(p=0.5) {
            self$p <- p
        },
        supp = function() { c(0, 1) },
        properties = function() {
            p <- self$p
            q <- 1.0 - p
            xlogx <- function(x) { ifelse(x > 0, x * log(x), 0) }
            list(succprob=p,
                 failprob=q,
                 mean=p,
                 median= if(p<=0.5) {0} else {1},
                 var=p * q,
                 skewness=(1 - 2*p) / sqrt(p*q),
                 kurtosis=(1 - 6*p*q) / (p*q),
                 entropy=-(xlogx(p) + xlogx(q)))
        },
        pdf = function(x, log=FALSE){ dbinom(x, 1, self$p, log=log) },
        cdf = function(x){ pbinom(x, 1, self$p) },
        quan = function(v){ qbinom(v, 1, self$p) }
    )
)
