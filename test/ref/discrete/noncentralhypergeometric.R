
FisherNoncentralHypergeometric <- R6Class("FisherNoncentralHypergeometric",
    inherit = DiscreteDistribution,
    public = list(
        names = c("m1", "m2", "n", "odds"),
        m1 = NA,
        m2 = NA,
        n = NA,
        odds = NA,
        initialize = function(ns, nf, n, o) {
            self$m1 <- ns
            self$m2 <- nf
            self$n <- n
            self$odds <- o
        },
        supp = function() {
            c( pmax(0, self$n - self$m2), pmin(self$m1, self$n) )
        },
        properties = function() {
            m1 <- self$m1
            m2 <- self$m2
            n <- self$n
            o <- self$odds
            list(mean = BiasedUrn::meanFNCHypergeo(m1, m2, n, o, precision=1e-16),
                 var = BiasedUrn::varFNCHypergeo(m1, m2, n, o, precision=1e-16),
                 mode = BiasedUrn::modeFNCHypergeo(m1, m2, n, o))
        },
        pdf = function(x, log=FALSE) {
            p <- BiasedUrn::dFNCHypergeo(
                x, self$m1, self$m2, self$n, self$odds, precision=1e-16)
            if (log) { log(p) } else { p }
        },
        cdf = function(x) {
            BiasedUrn::pFNCHypergeo(
                x, self$m1, self$m2, self$n, self$odds, precision=1e-16)
        },
        quan = function(v) {
            BiasedUrn::qFNCHypergeo(
                v, self$m1, self$m2, self$n, self$odds)
        }
    )
)

WalleniusNoncentralHypergeometric <- R6Class("WalleniusNoncentralHypergeometric",
    inherit = DiscreteDistribution,
    public = list(
        names = c("m1", "m2", "n", "odds"),
        m1 = NA,
        m2 = NA,
        n = NA,
        odds = NA,
        initialize = function(ns, nf, n, o) {
            self$m1 <- ns
            self$m2 <- nf
            self$n <- n
            self$odds <- o
        },
        supp = function() {
            c( pmax(0, self$n - self$m2), pmin(self$m1, self$n) )
        },
        properties = function() {
            m1 <- self$m1
            m2 <- self$m2
            n <- self$n
            o <- self$odds
            list(mean = BiasedUrn::meanWNCHypergeo(m1, m2, n, o, precision=1e-16),
                 var = BiasedUrn::varWNCHypergeo(m1, m2, n, o, precision=1e-16),
                 mode = BiasedUrn::modeWNCHypergeo(m1, m2, n, o))
        },
        pdf = function(x, log=FALSE) {
            p <- BiasedUrn::dWNCHypergeo(
                x, self$m1, self$m2, self$n, self$odds, precision=1e-16)
            if (log) { log(p) } else { p }
        },
        cdf = function(x) {
            BiasedUrn::pWNCHypergeo(
                x, self$m1, self$m2, self$n, self$odds, precision=1e-16)
        },
        quan = function(v) {
            BiasedUrn::qWNCHypergeo(
                v, self$m1, self$m2, self$n, self$odds)
        }
    )
)
