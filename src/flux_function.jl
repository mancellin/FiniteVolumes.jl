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
LinearAlgebra.eigvals(f::LinearAdvectionFlux, w, n) = f.velocity' * n

function directional_splitting(f::LinearAdvectionFlux{T}) where T<:SVector{2}
    (LinearAdvectionFlux(SVector(f.velocity[1], zero(eltype(f.velocity)))),
     LinearAdvectionFlux(SVector(zero(eltype(f.velocity)), f.velocity[2])))
end

##############################

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
    R = SMatrix{2, 2}(n[1], -n[2], n[2], n[1])
    u_local = R * SVector(ux, uy)
    ϕu_local = SVector(h*u_local[1]^2 + h^2 * f.g/2, h*u_local[1]*u_local[2])
    ϕu = R' * ϕu_local
    return SVector(h*u_local[1], ϕu[1], ϕu[2])
end

