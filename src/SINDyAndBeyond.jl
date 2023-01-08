module SINDyAndBeyond

include("library.jl")
include("output.jl")

export sparsifyDynamics, discover

variables(X; iterate_by=eachrow) = length(iterate_by(X'))

# TODO: Add comments and docstring
"""
Arguments:
- Θ is our data transformed by our selected library of candidate functions
- Ẋ is the derivative in our data
- λ is the *sparsification knob*
"""
function sparsifydynamics(Θ, Ẋ; λ, kwargs...)
    # n may be passed via keyword arguments
    n = !(@isdefined n) ? variables(Ẋ; kwargs...) : n
    Ξ = Θ \ Ẋ

    # TODO: Rework as general optimizer concept
    for _ in 1:10
        small_indices = @. abs(Ξ) < λ
        Ξ[small_indices] .= 0

        for index in 1:n
            big_indices_in = @. !small_indices[:, index]
            Ξ[big_indices_in, index] .= Θ[:, big_indices_in] \ Ẋ[:, index]
        end
    end

    return Ξ
end

# TODO: return a custom struct
function discover(X, Ẋ, library::T; kwargs...) where {T<:Library}
    Θ = library(X)
    Ξ = sparsifydynamics(Θ, Ẋ; kwargs...)
    return Ξ
end

end