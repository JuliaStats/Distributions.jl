# A R script to generate reference values for testing distributions
#
#   This script will take the following two list of distribution
#   entries as input:
#
#       - continuous_test.lst
#       - discrete_test.lst
#
#   The generated results will be written to the following files,
#   in JSON format, respectively
#
#       - continuous_test.json
#       - discrete_test.json
#
#   To run this script:
#
#       Rscript gendref.R
#
#   Note: this scripts depends on a number of R libraries for
#         file I/O and the computation of statistical functions.
#

library(stringr)
source("rdistributions.R")

options(
    error = function() {
        traceback(2)
        q()
    }
)

########################################
#
#  Parse distributions
#
########################################

read.entries <- function(filename) {
    # Read a list of entries from a given file
    lst <- read.table(filename, header=FALSE,
        sep="\t", blank.lines.skip = TRUE, comment.char="#")
    as.vector(lst$V1)
}

parse.entry <- function(entry) {
    # Parse an entry into a distribution name, and
    # a vector of argument values

    lb <- str_locate(entry, "\\(")[1]
    rb <- str_locate(entry, "\\)")[1]
    stopifnot(!is.na(lb), lb > 1)
    stopifnot(!is.na(rb), rb > lb)

    name <- str_sub(entry, 1, lb-1)
    if (rb == lb + 1) {
        args <- NULL
    } else {
        argstr <- entry %>% str_sub(lb+1, rb-1) %>% str_trim
        sl <- str_length(argstr)
        if (sl == 0) {
            args <- NULL
        } else {
            terms <- str_split(argstr, "\\s*,\\s*")[[1]]
            args <- as.numeric(terms)
        }
    }
    list(name=name, args=args)
}


get.distr <- function(entry) {
    # Get a distribution object based on a given entry
    parsed <- parse.entry(entry)
    dname <- parsed$name
    dargs <- parsed$args

    # obtain distribution generator
    dclass <- get(dname)

    # construct distribution
    nargs <- length(dargs)

    if (nargs == 0) {
        dclass$new()
    } else if (nargs == 1) {
        dclass$new(dargs[1])
    } else if (nargs == 2) {
        dclass$new(dargs[1], dargs[2])
    } else if (nargs == 3) {
        dclass$new(dargs[1], dargs[2], dargs[3])
    } else if (nargs == 4) {
        dclass$new(dargs[1], dargs[2], dargs[3], dargs[4])
    } else {
        stop(paste("Too many arguments for distribution", dname))
    }
}

########################################
#
#  Point wise evaluation
#
########################################

eval.samples <- function(distr) {
    if (distr$is.discrete) {
        su <- distr$supp()
        dmin <- su[1]
        dmax <- su[2]
        vmin <- if (is.finite(dmin)) { dmin } else { distr$quan(0.01) }
        vmax <- if (is.finite(dmax)) { dmax } else { distr$quan(0.99) }

        if (vmax - vmin + 1 <= 10) {
            seq(vmin, vmax)
        } else {
            xs <- unique(round(distr$quan(seq(0.1, 0.9, 0.1))))
            if (vmin < xs[1]) {
                xs <- c(vmin, xs)
            }
            if (vmax > xs[length(xs)]) {
                xs <- c(xs, vmax)
            }
            xs
        }
    } else {
        distr$quan(seq(0.1, 0.9, 0.1))
    }
}

eval.points <- function(distr) {
    xs <- eval.samples(distr)
    list(x  = xs,
         pd = distr$pdf(xs),
         lp = distr$pdf(xs, log=TRUE),
         cp = distr$cdf(xs))
}

eval.quans <- function(distr) {
    qs <- c(0.10, 0.25, 0.50, 0.75, 0.90)
    list(q = qs,
         x = distr$quan(qs))
}


########################################
#
#  JSON output
#
########################################

json.str <- function(val) {
    # convert a value to json string
    v <- val
    if (is.numeric(val)) {
        if (is.nan(val)) {
            v <- "nan"
        } else if (!is.finite(val)) {
            v <- ifelse(val > 0, "inf", "-inf")
        }
    }

    if (is.numeric(v)) {
        if (v == floor(v)) {
            sprintf("%g", v)
        } else {
            sprintf("%.15g", v)
        }
    } else {
        sprintf("\"%s\"", v)
    }
}

cat.attribute <- function (prefix, name, val, last=FALSE) {
    s <- sprintf("\"%s\": %s", name, json.str(val))
    if (last) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

cat.point <- function (prefix, points, i) {
    n <- length(points$x)
    x <- points$x[i]
    p <- points$pd[i]
    lp <- points$lp[i]
    cd <- points$cp[i]

    s <- sprintf("{ \"x\": %s, \"pdf\": %s, \"logpdf\": %s, \"cdf\": %s }",
        x, json.str(p), json.str(lp), json.str(cd))

    if (i == n) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

cat.quan <- function (prefix, quans, i) {
    n <- length(quans$q)
    q <- quans$q[i]
    x <- quans$x[i]
    s <- sprintf("{ \"q\": %.2f, \"x\": %s }", q, json.str(x))
    if (i == n) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

write.json <- function(info, last=FALSE) {
    cat("{\n")
    cat.attribute("  ", "expr", info$expr)
    cat.attribute("  ", "dtype", info$dtype)
    cat.attribute("  ", "minimum", info$supp[1])
    cat.attribute("  ", "maximum", info$supp[2])

    # output properties
    props <- info$props
    cat("  \"properties\": {\n")
    i <- 0
    n <- length(props)
    for (pn in names(props)) {
        i <- i + 1
        cat.attribute("    ", pn, props[[pn]], last=(i==n))
    }
    cat("  },\n")

    # output points
    points <- info$points
    n <- length(points$x)
    if (n > 0) {
        cat("  \"points\": [\n")
        for (i in 1:n) {
            cat.point("    ", points, i)
        }
        cat("  ],\n")
    } else {
        cat("  \"points\": [],\n")
    }

    # output quantile points
    quans <- info$quans
    n <- length(quans$q)
    if (n > 0) {
        cat("  \"quans\": [\n")
        for (i in 1:n) {
            cat.quan("    ", quans, i)
        }
        cat("  ]\n")
    } else {
        cat("  \"quans\": []\n")
    }

    # closing
    if (last) {
        cat("}\n")
    } else {
        cat("},\n")
    }
}


########################################
#
#  Main program
#
########################################

do.main <- function(lstname) {
    # read a list of entries from file <lstname>.lst
    cat("For list", lstname, "\n")
    cat("-------------------------------\n")
    lstfile <- sprintf("%s.lst", lstname)
    entries <- read.entries(lstfile)

    # derive information
    infolist <- list()
    n <- 0
    for (e in entries) {
        distr <- get.distr(e)
        dtype <- class(distr)[1]
        cat("On ", e, " --> ", dtype, ": ", sep="")
        for (sn in distr$names) {
            cat(sn, "=", distr[[sn]], " ", sep="")
        }
        cat("\n")
        n <- n + 1
        info <- list(
            dtype=class(distr)[1],
            expr=e,
            supp=distr$supp(),
            props=distr$properties(),
            points=eval.points(distr),
            quans=eval.quans(distr))
        infolist[[n]] <- info
    }

    # output
    outfile = sprintf("%s.ref.json", lstname)
    sink(outfile)
    cat("[\n")
    for (i in 1:n) {
        write.json(infolist[[i]], last=(i==n))
    }
    cat("]\n")
    sink()

    cat("\n")
}

do.main("discrete_test")
do.main("continuous_test")
