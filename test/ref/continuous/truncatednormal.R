
TruncatedNormal <- R6Class("TruncatedNormal",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma", "a", "b"),
        mu = NA,
        sigma = NA,
        a = NA,
        b = NA,
        initialize = function(u, s, a, b) {
            self$mu <- u
            self$sigma <- s
            self$a <- a
            self$b <- b
        },
        dtype = "Distributions.Truncated{<:Normal}",
        expr = function() {
            str <- paste0("truncated(Normal(", self$mu, ", ", self$sigma, ")")
            if (!is.infinite(self$a)) {
                str <- paste0(str, ", lower=", self$a)
            }
            if (!is.infinite(self$b)) {
                str <- paste0(str, ", upper=", self$b)
            }
            paste0(str, ")")
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            a <- self$a
            b <- self$b
            za <- (a - u) / s
            zb <- (b - u) / s
            Z <- pnorm(zb) - pnorm(za)
            pa <- dnorm(za)
            pb <- dnorm(zb)
            v1 <- (ifelse(pa == 0, 0, za * pa) -
                   ifelse(pb == 0, 0, zb * pb)) / Z
            list(mode = pmin(pmax(u, a), b),
                 mean = u + (pa - pb) / Z * s,
                 var = s^2 * (1 + v1 - ((pa - pb) / Z)^2),
                 entropy = (log(2*pi) + 1) / 2 + log(s) + log(Z) + v1 / 2)
        },
        pdf = function(x, log=FALSE) { extraDistr::dtnorm(x, self$mu, self$sigma, self$a, self$b, log=log) },
        cdf = function(x) { extraDistr::ptnorm(x, self$mu, self$sigma, self$a, self$b) },
        quan = function(v) { extraDistr::qtnorm(v, self$mu, self$sigma, self$a, self$b) }
    )
)
