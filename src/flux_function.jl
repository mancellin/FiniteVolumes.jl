using ForwardDiff

# Linear
struct LinearAdvectionFlux{V}
    velocity::V
end

LinearAlgebra.eigvals(f::LinearAdvectionFlux, w, n) = f.velocity'*n
(f::LinearAdvectionFlux)(w, n) = w*(f.velocity'*n)

# Wave equation
struct Wave1DFlux{T}
    velocity::T
end

(f::Wave1DFlux)(w, n) = f.velocity*SVector{2}(w[2], w[1])*n
jacobian(f::Wave1DFlux, w, n) = f.velocity*SMatrix{2, 2}(0.0, 1.0, 1.0, 0.0)*n
LinearAlgebra.eigen(f::Wave1DFlux, w, n) = SVector{2}(-1.0, 1.0)*n, SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n
# jacobian(f::Wave1DFlux, w::SVector{2, T}, n) where T = f.velocity*SMatrix{2, 2}(zero(T)/oneunit(T), one(T), one(T), zero(T)/oneunit(T))*n
# LinearAlgebra.eigen(f::Wave1DFlux, w::SVector{2, T}, n) where T = SVector{2}(-one(T), one(T))*n, SMatrix{2, 2}(-0.707107, 0.707107, 0.707107, 0.707107)*n

# Generic
struct FluxFunction{T, D, F}
    func::F
end
FluxFunction{T, D}(f) where {T, D} = FluxFunction{T, D, typeof(f)}(f)

(f::FluxFunction{T, 1})(w, n) where T = f.func(w)*n
(f::FluxFunction)(w, n) = f.func(w)'*n
LinearAlgebra.eigvals(f::FluxFunction{<:Number}, w, n) = ForwardDiff.derivative(v -> f(v, n), w)

jacobian(f::FluxFunction, w, n) = ForwardDiff.jacobian(v -> f(v, n), w)
LinearAlgebra.eigvals(f::FluxFunction{T}, w, n) where {T <: AbstractArray} = eigen(jacobian(f, w, n)).values .|> real
function LinearAlgebra.eigen(f::FluxFunction{T}, w, n) where {T <: AbstractArray}
    λc, Lc = eigen(jacobian(f, w, n))
    return real.(λc), real.(Lc)
end
