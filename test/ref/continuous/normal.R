
Normal <- R6Class("Normal",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u=0, s=1) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            list(mean=u,
                 var=s^2,
                 location=u,
                 scale=s,
                 skewness=0,
                 kurtosis=0,
                 entropy=(log(2 * pi) + 1 + log(s^2))/2)
        },
        pdf = function(x, log=FALSE) { dnorm(x, self$mu, self$sigma, log=log) },
        cdf = function(x) { pnorm(x, self$mu, self$sigma) },
        quan = function(v) { qnorm(v, self$mu, self$sigma ) }
    )
)

NormalCanon = list(
    new = function(c1=0, c2=1) {
        Normal$new(c1/c2, 1/sqrt(c2))
    }
)
