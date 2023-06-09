library("LindleyR")

Lindley <- R6Class("Lindley",
                   inherit=ContinuousDistribution,
                   public=list(names=c("theta"),
                               theta=NA,
                               initialize=function(theta=1) { self$theta <- theta },
                               supp=function() { c(0, Inf) },
                               properties=function() { list() },
                               pdf=function(x, log=FALSE) { dlindley(x, self$theta, log=log) },
                               cdf=function(x) { plindley(x, self$theta) },
                               quan=function(x) { qlindley(x, self$theta) }))
