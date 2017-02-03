
Binomial <- R6Class("Binomial",
    inherit = DiscreteDistribution,
    public = list(
        names = c("n", "p"),
        n = NA,
        p = NA,
        initialize = function(n=1, p=0.5) {
            self$n <- n
            self$p <- p
        },
        supp = function() { c(0, self$n) },
        properties = function() {
            n <- self$n
            p <- self$p
            q <- 1.0 - p
            list(succprob=p,
                 failprob=q,
                 ntrials=n,
                 mean=n * p,
                 median=round(n * p),
                 var=n * p * q,
                 skewness=(q - p) / sqrt(n*p*q),
                 kurtosis=(1 - 6*p*q) / (n*p*q))
        },
        pdf = function(x, log=FALSE) { dbinom(x, self$n, self$p, log=log) },
        cdf = function(x) { pbinom(x, self$n, self$p) },
        quan = function(v) { qbinom(v, self$n, self$p) }
    )
)
