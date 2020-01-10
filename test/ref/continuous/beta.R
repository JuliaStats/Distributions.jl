
Beta <- R6Class("Beta",
    inherit = ContinuousDistribution,
    public = list(
        names = c("alpha", "beta"),
        alpha = NA,
        beta = NA,
        initialize = function(a=1, b=a) {
            self$alpha <- a
            self$beta <- b
        },
        supp = function() { c(0.0, 1.0) },
        properties = function() {
            a <- self$alpha
            b <- self$beta
            skew <- 2 * (b - a) * sqrt(a + b + 1) / (a + b + 2) / sqrt(a * b)
            kurt.num <- 6 * ((a - b)^2 * (a + b + 1) - a * b * (a + b + 2))
            kurt.den <- a * b * (a + b + 2) * (a + b + 3)
            ent <- lbeta(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) +
                   (a + b - 2) * digamma(a + b)
            list(mean=a / (a + b),
                 meanlogx=digamma(a) - digamma(a + b),
                 var=(a * b) / (a + b)^2 / (a + b + 1.0),
                 varlogx=trigamma(a) - trigamma(a + b),
                 skewness=skew,
                 kurtosis=kurt.num / kurt.den,
                 entropy=ent)
        },
        pdf = function(x, log=FALSE){ dbeta(x, self$alpha, self$beta, log=log) },
        cdf = function(x) { pbeta(x, self$alpha, self$beta) },
        quan = function(v) { qbeta(v, self$alpha, self$beta) }
    )
)
