using LinearAlgebra: I

"""
    ScalarLinearAdvection{N, T, D}

where
	N: number of fields
	D: dimension
	T: data type

Advection of N fields with the same constant and uniform velocity.
"""
struct ScalarLinearAdvection{N, D, T} <: AbstractModel
    velocity::SVector{D, T}
end

ScalarLinearAdvection(N, v) = ScalarLinearAdvection{N, length(v), eltype(v)}(SVector{length(v), eltype(v)}(v...))
ScalarLinearAdvection(v) = ScalarLinearAdvection(1, v)

# Legacy
NScalarLinearAdvection(args...) = ScalarLinearAdvection(args...)

function directional_splitting(s::ScalarLinearAdvection{N, 2, T}) where {N, T}
    sx, sy = s.velocity
    [ScalarLinearAdvection{N, 2, T}([sx, zero(T)]), ScalarLinearAdvection{N, 2, T}([zero(T), sy])]
end

nb_vars(m::ScalarLinearAdvection{N, D, T}) where {N, D, T} = N
Base.eltype(m::ScalarLinearAdvection{N, D, T}) where {N, D, T} = T
nb_dims(m::ScalarLinearAdvection{N, D, T}) where {N, D, T} = D

w_names(m::ScalarLinearAdvection{1, D, T}) where {D, T} = (:α,)
w_names(m::ScalarLinearAdvection{N, D, T}) where {N, D, T} = Tuple(Symbol("α_$i") for i in 1:N)

rotate_model(m::ScalarLinearAdvection, rotation_matrix) = typeof(m)(rotation_matrix' * m.velocity)

function normal_flux(m::ScalarLinearAdvection{N, D, T}, w, wsupp) where {N, D, T}
    return typeof(w)(w * m.velocity[1])
    # Only the first coordinate in the frame of the interface, i.e. the normal vector
end

eigenvalues(m::ScalarLinearAdvection{N, D, T}, w, wsupp) where {N, D, T} = @SVector fill(m.velocity[1], N)
left_eigenvectors(m::ScalarLinearAdvection{N, D, T}, w, wsupp) where {N, D, T} = SMatrix{N, N, T}(I)
right_eigenvectors(m::ScalarLinearAdvection{N, D, T}, w, wsupp) where {N, D, T} = SMatrix{N, N, T}(I)
