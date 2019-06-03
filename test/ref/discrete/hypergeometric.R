
Hypergeometric <- R6Class("Hypergeometric",
    inherit = DiscreteDistribution,
    public = list(
        names = c("ns", "nf", "n"),
        ns = NA,
        nf = NA,
        n = NA,
        initialize = function(ns, nf, n) {
            self$ns <- ns
            self$nf <- nf
            self$n <- n
        },
        supp = function(){
            N <- self$ns + self$nf
            K <- self$ns
            n <- self$n
            c(max(0, n+K-N), min(n, K))
        },
        properties = function() {
            N <- self$ns + self$nf
            K <- self$ns
            n <- self$n
            mean.val <- n * (K / N)
            var.val  <- n * (K / N) * ((N-K) / N) * ((N - n) / (N - 1))
            skew.val <- ((N - 2*K) * sqrt(N - 1) * (N - 2*n)) /
                (sqrt(n * K * (N - K) * (N - n)) * (N - 2))
            kurt.num <- (N-1) * N^2 * (N * (N+1)- 6 * K * (N-K) - 6 * n * (N-n)) +
                6 * n * K * (N-K) * (N-n) * (5*N-6)
            kurt.den <- n * K * (N - K) * (N - n) * (N - 2) * (N - 3)
            kurt.val <- kurt.num / kurt.den
            list(mean=mean.val,
                 var=var.val,
                 skewness=skew.val,
                 kurtosis=kurt.val)
        },
        pdf = function(x, log=FALSE){ dhyper(x, self$ns, self$nf, self$n, log=log) },
        cdf = function(x){ phyper(x, self$ns, self$nf, self$n) },
        quan = function(v){ qhyper(v, self$ns, self$nf, self$n) }
    )
)
