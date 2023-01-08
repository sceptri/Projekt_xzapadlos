export MonomialLibrary, PolynomialLibrary, CustomLibrary

abstract type Library end

function (library::Library)(X; iterate_by=eachrow, kwargs...)
    # `iterate_by(X)` is an iterator that returns temporal states of the system,
    # therefore `iterate_by(X')` returns all temporal data of each variable per iteration 
    Θ = reshape.([func(X) for func in library], length(iterate_by(X)), 1)
    return hcat(Θ...)
end

abstract type PredefinedLibrary <: Library end

eachstate(func, X; iterate_by=eachrow) = [func(iteration) for iteration in iterate_by(X)]

# TODO: rework MonomialLibrary using `deconstruct_state`
struct MonomialLibrary{T<:Integer} <: PredefinedLibrary
    degree::T
    variables::T
end

Base.length(library::MonomialLibrary) = (library.degree + 1) * library.variables
Base.iterate(library::MonomialLibrary; kwargs...) = Base.iterate(library, (0, 0); kwargs...)

function Base.iterate(library::MonomialLibrary, state; kwargs...)
    power, var = state
    var += 1
    if var > library.variables
        power += 1
        var = 1
    end
    if power > library.degree
        return nothing
    end

    func = X -> eachstate(system_state -> system_state[var]^power, X; kwargs...)
    return (func, (power, var))
end

struct PolynomialLibrary{T<:Integer} <: PredefinedLibrary
    degree::T
    variables::T
end

"""
I recommend drawing it to really get what's happening.

For 2 variables x, y and polynomial up to 1st order in each variable
- x stays the same for 2 iterations
- y stays the same for 2^0 iterations, repeats every 2 iterations
"""
deconstruct_state(library::PolynomialLibrary, state::Integer) =
    [((state - 1) ÷ ((library.degree + 1)^index)) % (library.degree + 1) for index in reverse(0:(library.variables-1))]

Base.length(library::PolynomialLibrary) = (library.degree + 1)^library.variables
Base.iterate(library::PolynomialLibrary; kwargs...) = Base.iterate(library, 0; kwargs...)

function Base.iterate(library::PolynomialLibrary, state; kwargs...)
    state += 1

    if state > length(library)
        return nothing
    end

    power_vector = deconstruct_state(library, state)
    func = X -> eachstate(system_state -> prod(system_state .^ power_vector), X; kwargs...)
    return (func, state)
end

struct CustomLibrary{T<:AbstractArray} <: Library
    functions::T
end

Base.length(library::CustomLibrary) = length(library.functions)
Base.iterate(library::CustomLibrary; kwargs...) = Base.iterate(library, 0; kwargs...)

function Base.iterate(library::CustomLibrary, index; kwargs...)
    index += 1
    if index > length(library.functions)
        return nothing
    end

    func = X -> eachstate(system_state -> library.functions[index](system_state), X; kwargs...)
    return (func, index)
end
