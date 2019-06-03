
Arcsine <- R6Class("Arcsine",
    inherit = ContinuousDistribution,
    public = list(
        names = c("a", "b"),
        a = NA,
        b = NA,
        rd = NA,  # R distr object
        initialize = function(a1=NA, a2=NA) {
            if (is.na(a1)) {
                a <- 0; b <- 1
            } else if (is.na(a2)) {
                a <- 0; b <- a1
            } else {
                a <- a1; b <- a2
            }
            self$a <- a
            self$b <- b
            self$rd <- distr::Arcsine() * ((b-a)/2) + ((a+b)/2)
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            s <- self$b - self$a
            list(location=self$a,
                 scale=s,
                 mean=(self$a + self$b) * 0.5,
                 var=1/8 * s^2,
                 skewness=0,
                 kurtosis=-1.5,
                 median=(self$a + self$b) * 0.5,
                 entropy=log(pi/4) + log(s))
        },
        pdf = function(x, log=FALSE) { distr::d(self$rd)(x, log=log) },
        cdf = function(x) { distr::p(self$rd)(x) },
        quan = function(v) { distr::q(self$rd)(v) }
    )
)
