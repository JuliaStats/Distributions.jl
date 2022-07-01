
Uniform <- R6Class("Uniform",
    inherit = ContinuousDistribution,
    public = list(
        names = c("a", "b"),
        a = NA,
        b = NA,
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
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            a <- self$a
            b <- self$b
            list(location = a,
                 scale = b - a,
                 mean = (a + b) / 2,
                 median = (a + b) / 2,
                 var = (b - a)^2 / 12,
                 skewness = 0,
                 kurtosis = -6 / 5,
                 entropy = log(b - a))
        },
        pdf = function(x, log=FALSE) { dunif(x, self$a, self$b, log=log) },
        cdf = function(x) { punif(x, self$a, self$b) },
        quan = function(v) { qunif(v, self$a, self$b) }
    )
)
