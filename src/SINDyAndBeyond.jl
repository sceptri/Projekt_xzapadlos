module SINDyAndBeyond

variables(X; iterate_by=eachrow) = length(iterate_by(X'))
states(X; iterate_by=eachrow) = length(iterate_by(X))

include("library.jl")
include("optimizer.jl")
include("output.jl")

export discover, L₂

# TODO: return a custom struct
function discover(X, Ẋ, library::T, optimizer::TT; kwargs...) where {T<:Library,TT<:RegressionOptimizer}
    Θ = library(X; kwargs...)
    Ξ = optimizer(Θ, Ẋ; kwargs...)
    return Ξ
end

function discover(X, library::T, optimizer::TT;
    timespan=(0, 10),
    u₀=ones(variables(X)),
    saveat=0.1,
    kwargs...
) where {T<:Library,TT<:DiffEqOptimizer}
    Θ_problem = diffeq_problem(library, X, timespan, u₀; kwargs...)
    Ξ = optimizer(Θ_problem, X, saveat; kwargs...)
    return Ξ
end

L₂(truth::AbstractArray, compare::AbstractArray) = sqrt.(sum((compare .- truth) .^ 2; dims=2))

end