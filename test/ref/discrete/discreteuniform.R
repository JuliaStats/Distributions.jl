
DiscreteUniform <- R6Class("DiscreteUniform",
    inherit = DiscreteDistribution,
    public = list(
        a = NA,
        b = NA,
        s = NA,
        names = c("a", "b"),
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
            self$s <- b - a + 1
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            a <- self$a
            b <- self$b
            s <- self$s
            list(span=s,
                 probval=1/s,
                 mean=(a + b)/2,
                 median=floor((a + b)/2),
                 var=(s^2 - 1)/12,
                 skewness=0,
                 kurtosis=-(6 * (s^2 + 1))/(5 * (s^2 - 1)),
                 entropy=log(s))
        },
        pdf = function(x, log=FALSE) {
            a <- self$a
            b <- self$b
            s <- self$s
            t <- x >= a & x <= b
            if (log) {
                ifelse(t, -log(s), -Inf)
            } else {
                ifelse(t, 1 / s, 0.0)
            }
        },
        cdf = function(x) {
            a <- self$a
            b <- self$b
            r <- (x - a + 1) / (b - a + 1)
            pmin(pmax(r, 0), 1)
        },
        quan = function(v) {
            a <- self$a
            b <- self$b
            cv <- pmin(pmax(v, 0), 1)
            ifelse(cv == 0, a, a - 1 + ceiling((b - a + 1) * cv))
        }
    )
)
