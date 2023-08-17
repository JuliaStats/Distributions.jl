Kumaraswamy <- R6Class("Kumaraswamy",
                       inherit=ContinuousDistribution,
                       public=list(names=c("a", "b"),
                                   a=NA,
                                   b=NA,
                                   initialize=function(a=1, b=1) {
                                       self$a <- a
                                       self$b <- b
                                   },
                                   supp=function() { c(0, 1) },
                                   properties=function() { list() },
                                   pdf=function(x, log=FALSE) { dkumar(x, self$a, self$b, log=log) },
                                   cdf=function(x) { pkumar(x, self$a, self$b) },
                                   quan=function(x) { qkumar(x, self$a, self$b) }))
