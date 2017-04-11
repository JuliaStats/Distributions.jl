
BetaBinomial <- R6Class("BetaBinomial",
    inherit = DiscreteDistribution,
    public = list(
        names = c("n", "alpha", "beta"),
        n = NA,
        alpha = NA,
        beta = NA,
        probs = NA,
        initialize = function(n, a, b) {
            self$n <- n
            self$alpha <- a
            self$beta <- b
            self$probs <- dbbinom(0:n, self$n, self$alpha, self$beta)
        },
        supp = function() { c(0, self$n) },
        properties = function() {
            n <- self$n
            a <- self$alpha
            b <- self$beta
            u <- a + b
            v1 <- n * a * b * (a+b+n)
            kur.orig <- u^2 * (1+u) / (v1 * (u+2) * (u+3)) * (
               u*(u-1+6*n) + 3*a*b*(n-2) + 6*n^2 -
               3*a*b*n*(6-n)/u - 18*a*b*n^2/u^2)
            list(ntrials = n,
                 mean = n * a / u,
                 var = v1 / u^2 / (u+1),
                 skewness = (u+2*n)*(b-a)/(u+2) * sqrt((u+1)/v1),
                 kurtosis = kur.orig-3)
        },
        pdf = function(x, log=FALSE){ dbbinom(x, self$n, self$alpha, self$beta, log=log) },
        cdf = function(x) { pbbinom(x, self$n, self$alpha, self$beta) },
        quan = function(v) { qcat(v, self$probs) - 1 }
    )
)
