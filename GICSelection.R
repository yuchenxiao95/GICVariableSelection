#' Perform GIC Variable Selection using Julia
#'
#' This function wraps a Julia function to perform GIC-based variable selection
#' in parallel.
#'
#' @param X A numeric matrix of input features.
#' @param Y A numeric vector of target values.
#' @param i An integer vector of initial feature indices.
#' @param Calculate_GIC A Julia function to compute GIC using the full inverse.
#' @param Calculate_GIC_short A Julia function to compute GIC incrementally.
#' @param debug Logical. If TRUE, prints debugging information. Default is FALSE.
#'
#' @return A list with:
#'   \item{GIC_list}{A numeric vector of GIC values.}
#'   \item{GIC_coeff}{A list of subsets corresponding to non-zero coefficients.}
#' @export
#' @examples
#' \dontrun{
#' julia_setup()
#' result <- GICSelection(X, Y, i, Calculate_GIC, Calculate_GIC_short, debug = FALSE)
#' }
GICSelection <- function(X, Y, i, Calculate_GIC, Calculate_GIC_short, debug = FALSE) {
  # Ensure Julia is set up
  JuliaCall::julia_setup()
  
  # Load the Julia function from inst/julia/
  JuliaCall::julia_source(system.file("julia", "GIC_Variable_Selection_Parallel.jl", package = "GICModelSelection"))
  
  # Call the Julia function
  result <- JuliaCall::julia_call("GIC_Variable_Selection_Parallel", 
                                  as.matrix(X), 
                                  as.vector(Y), 
                                  as.integer(i), 
                                  Calculate_GIC, 
                                  Calculate_GIC_short,
                                  debug = debug)
  
  return(result)
}
