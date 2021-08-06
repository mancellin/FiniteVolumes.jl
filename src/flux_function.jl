using ForwardDiff

abstract type AbstractFlux end

jacobian(f::AbstractFlux, w, n) = ForwardDiff.jacobian(v -> f(v, n), w)
LinearAlgebra.eigvals(f::AbstractFlux, w, n) = eigen(jacobian(f, w, n)).values .|> real
function LinearAlgebra.eigen(f::AbstractFlux, w, n)
    λc, Rc = eigen(jacobian(f, w, n))
    return real.(λc), real.(Rc)
end

##############################

struct LinearAdvectionFlux{V} <: AbstractFlux
    velocity::V
end
LinearAdvectionFlux(v::AbstractArray) = LinearAdvectionFlux{SVector{length(v), eltype(v)}}(SVector(v...))

(f::LinearAdvectionFlux)(w, n) = w * (f.velocity' * n)
jacobian(f::LinearAdvectionFlux, w, n) = f.velocity' * n
LinearAlgebra.eigvals(f::LinearAdvectionFlux, w, n) = f.velocity' * n

##############################

# ! SCALAR FLUX FUNCTION ONLY
struct FluxFunction{T, D, F} <: AbstractFlux
    # T = datatype
    # D = space dimension
    func::F
end
FluxFunction{T, D}(f) where {T, D} = FluxFunction{T, D, typeof(f)}(f)

(f::FluxFunction{T, 1})(w::T, n) where T = f.func(w) * n
(f::FluxFunction)(w, n) = (f.func(w)' * n)'

jacobian(f::FluxFunction{T, D}, w::T, n) where {T <: Number, D} = ForwardDiff.derivative(v -> f(v, n), w)
LinearAlgebra.eigvals(f::FluxFunction{T, D}, w::D, n) where {T <: Number, D} = jacobian(f, w, n)

##############################

struct ShallowWater{T} <: AbstractFlux
    g::T
end
ShallowWater(; g=9.81) = ShallowWater{typeof(g)}(g)

function (f::ShallowWater)(v::SVector{2}, n::Union{Number, Scalar, SVector{1}})
    h, hu = v[1], v[2]
    SVector(hu * n, hu^2/h + h^2 * f.g/2) * n
end

function (f::ShallowWater)(v::SVector{3}, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux*n[1] + uy*n[2]
    return SVector(h*un, h*ux*un + h^2*n[1]*f.g/2, h*uy*un + h^2*n[2]*f.g/2)
end

function jacobian(f::ShallowWater, v, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux * n[1] + uy * n[2]
    g = f.g
    @SMatrix [0                 n[1]        n[2];
              -ux*un+g*h*n[1]   un+ux*n[1]  ux*n[2];
              -uy*un+g*h*n[2]   uy*n[1]     un+uy*n[2]]
end

function LinearAlgebra.eigvals(f::ShallowWater, v, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux * n[1] + uy * n[2]
    g = f.g
    c = sqrt(g*h)
    return SVector(un - c, un, un + c)
end

function LinearAlgebra.eigen(f::ShallowWater, v, n::SVector{2})
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    un = ux * n[1] + uy * n[2]
    ut = -ux * n[2] + uy * n[1]
    g = f.g
    c = sqrt(g*h)
    vals = SVector(un - c, un, un + c)
    vects = 0.5 .* @SMatrix [1.0                    0.0   1.0;
                             n[1]*(un-c) + n[2]*ut  n[2]  n[1]*(c+un) + n[2]*ut;
                             n[1]*ut + n[2]*(un-c)  n[1]  n[1]*ut + n[2]*(c+un)]
    return vals, vects 
end
