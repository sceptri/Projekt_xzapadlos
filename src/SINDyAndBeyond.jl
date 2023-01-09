module SINDyAndBeyond

include("library.jl")
include("optimizer.jl")
include("output.jl")

export discover, L₂

variables(X; iterate_by=eachrow) = length(iterate_by(X'))

# TODO: return a custom struct
function discover(X, Ẋ, library::T, optimizer::TT; kwargs...) where {T<:Library,TT<:Optimizer}
    Θ = library(X)
    Ξ = optimizer(Θ, Ẋ; kwargs...)
    return Ξ
end

L₂(truth::AbstractArray, compare::AbstractArray) = sqrt.(sum((compare .- truth) .^ 2; dims=2))

end