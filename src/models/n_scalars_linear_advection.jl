using LinearAlgebra: I

struct NScalarLinearAdvection{N, T, D} <: AbstractModel
	# N: number of fields
	# T: float type
	# D: dimension
    velocity::SVector{D, T}
end

NScalarLinearAdvection(N, v) = NScalarLinearAdvection{N, typeof(v), 1}(SVector{1, typeof(v)}(v))

Base.eltype(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = T
nb_dims(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = D
nb_vars(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = N
w_names(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = Tuple(Symbol("α_$i") for i in 1:N)

rotate_model(m::NScalarLinearAdvection{N, T, D}, rotation_matrix) where {N, T, D}= NScalarLinearAdvection{N, T, D}(rotation_matrix * m.velocity)

function normal_flux(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D}
    return SVector{N, T}(w * m.velocity[1])
    # Only the first coordinate in the frame of the interface, i.e. the normal vector
end

@inline eigenvalues(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D} = @SVector fill(m.velocity[1], N)
@inline left_eigenvectors(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D} = SMatrix{N, N, T}(I)
@inline right_eigenvectors(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D} = SMatrix{N, N, T}(I)
