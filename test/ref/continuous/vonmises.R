
VonMises <- R6Class("VonMises",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "kappa"),
        mu = NA,
        kappa = NA,
        initialize = function(a1=NA, a2=NA) {
            if (is.na(a1)) {
                u <- 0; k <- 1
            } else if (is.na(a2)) {
                u <- 0; k <- a1
            } else {
                u <- a1; k <- a2
            }
            self$mu <- circular::circular(u, units='radian')
            self$kappa <- k
        },
        supp = function() { c(self$mu - self$kappa, self$mu + self$kappa) },
        properties = function() {
            u <- self$mu
            k <- self$kappa
            I0k <- besselI(k, 0)
            I1k <- besselI(k, 1)
            list(mean = u,
                 median = u,
                 mode = u,
                 var = 1 - I1k / I0k,
                 entropy = log(2*pi*I0k) - k * I1k/I0k)
        },
        pdf = function(x, log=FALSE) {
            circular::dvonmises(circular::circular(x, units='radian'),
                self$mu, self$kappa, log=log)
        },
        cdf = function(x) {
            circular::pvonmises(circular::circular(x, units='radian'),
                self$mu, self$kappa)
        },
        quan = function(v) {
            circular::qvonmises(v, self$mu, self$kappa)
        }
    )
)
