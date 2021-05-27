
AsymmetricLaplace <- R6Class("Asymmetriclaplace",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "lambda", "kappa"),
        mu = NA,
        lambda = NA,
        kappa = NA,
        initialize = function(u=0, l=1, k=1) {
            self$mu <- u
            self$beta <- l
            self$kappa <- k
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            l <- self$lambda
            k <- self$kappa
            list(location = u,
                 scale = l,
                 mean = u + (1-k^2) / (l * k),
                 median = ifelse(test=k > 1, yes=u + k / l * log((1 + k^2) / (2 * k^2)), no=u - 1 / (k * l) * log((1 + k^1) / 2)),
                 var = (1 + k^4) / (l^2 * k^2),
                 skewness = 2 * (1 - k^6) / (k^4 + 1)^(3 / 2),
                 kurtosis = 6 * (1 + k^8) / (1 + k^4)^2,
                 entropy = log(exp(1) * (1 + k) / (l * k))
        },
        pdf = function(x){ ifelse(test=x > self$mu, yes=self$lambda / (self$kappa + 1 / self$kappa) * exp(-self$lambda * self$kappa * (x - self$mu)), no=self$lambda / (self$kappa + 1 / self$kappa) * exp(self$lambda / self$kappa * (x - self$mu))) },
        cdf = function(x){ ifelse(test=x > self$mu, yes=1 - 1 / (1 + self$kappa^2) * exp(-self$lambda * self$kappa * (x - self$mu)), no=self$kappa^2 / (1 + self$kappa^2) * exp(self$lambda / self$kappa * (x - self$mu))) },
        quan = function(v){ ifelse(test=v > self$cdf(self$mu), yes=self$mu - (log(1 - v) + log(1 + self$kappa^2)) / (self$lambda * self$kappa), no=self$mu + (self$kappa / self$lambda) * (log(v) + log(1 + self$kappa^2) - 2 * log(self$kappa))) }
    )
)
