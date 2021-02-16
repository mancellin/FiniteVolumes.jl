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

(f::LinearAdvectionFlux)(w, n) = w * (f.velocity' * n)
LinearAlgebra.eigvals(f::LinearAdvectionFlux, w, n) = f.velocity' * n

function directional_splitting(f::LinearAdvectionFlux{T}) where T<:SVector{2}
    (LinearAdvectionFlux(SVector(f.velocity[1], zero(eltype(f.velocity)))),
     LinearAdvectionFlux(SVector(zero(eltype(f.velocity)), f.velocity[2])))
end

##############################

struct Wave1DFlux{T} <: AbstractFlux
    velocity::T
end

(f::Wave1DFlux)(w, n) = f.velocity*SVector{2}(w[2], w[1])*n

jacobian(f::Wave1DFlux, w, n) = f.velocity*SMatrix{2, 2}(0.0, 1.0, 1.0, 0.0)*n
LinearAlgebra.eigvals(f::Wave1DFlux, w, n) = SVector{2}(-1.0, 1.0)*n
LinearAlgebra.eigen(f::Wave1DFlux, w, n) = (eigvals(f, w, n), SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n)

# jacobian(f::Wave1DFlux, w::SVector{2, T}, n) where {T} = f.velocity * SMatrix{2, 2}(zero(T)/oneunit(T), one(T), one(T), zero(T)/oneunit(T)) * n
# LinearAlgebra.eigvals(f::Wave1DFlux, w::SVector{2, T}, n) where {T} = SVector{2}(-one(T), one(T)) * n
# LinearAlgebra.eigen(f::Wave1DFlux, w::SVector{2, T}, n) where {T} = (eigvals(f, w, n), SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n)

##############################

struct FluxFunction{T, D, F} <: AbstractFlux
    # T = datatype
    # D = space dimension
    func::F
end
FluxFunction{T, D}(f) where {T, D} = FluxFunction{T, D, typeof(f)}(f)

(f::FluxFunction{T, 1})(w, n) where T = f.func(w) * n
(f::FluxFunction)(w, n) = f.func(w)' * n

LinearAlgebra.eigvals(f::FluxFunction{<:Number}, w, n) = ForwardDiff.derivative(v -> f(v, n), w)

##############################

struct ShallowWater{T} <: AbstractFlux
    g::T
end

function (f::ShallowWater)(v::SVector{2}, n)
    h, hu = v[1], v[2]
    SVector(hu * n, hu^2/h + h^2 * f.g/2) * n
end
function (f::ShallowWater)(v::SVector{3}, n)
    h, ux, uy = v[1], v[2]/v[1], v[3]/v[1]
    R = SMatrix{2, 2}(n[1], -n[2], n[2], n[1])
    u_local = R * SVector(ux, uy)
    ϕu_local = SVector(h*u_local[1]^2 + h^2 * f.g/2, h*u_local[1]*u_local[2])
    ϕu = R' * ϕu_local
    return SVector(h*u_local[1], ϕu[1], ϕu[2])
end
