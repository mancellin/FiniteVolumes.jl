using LinearAlgebra: I

struct NScalarLinearAdvection{N, T, D}
	# N: number of fields
	# T: float type
	# D: dimension
    velocity::SVector{D, T}
end

NScalarLinearAdvection(N, v) = NScalarLinearAdvection{N, typeof(v), 1}(SVector{1, typeof(v)}(v))

Base.eltype(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = T
nb_dims(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = D
nb_vars(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = N
nb_vars_supp(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = 0

w_names(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = Tuple(Symbol("α_$i") for i in 1:N)
wsupp_names(m::NScalarLinearAdvection{N, T, D}) where {N, T, D} = Tuple([])

@inline rotate_model(m::NScalarLinearAdvection{N, T, D}, rotation_matrix) where {N, T, D}= NScalarLinearAdvection{N, T, D}(rotation_matrix * m.velocity)

@inline compute_wsupp(m::NScalarLinearAdvection, w) = SVector{0, Float64}()
@inline rotate_state(w, wsupp, m::NScalarLinearAdvection, rotation_matrix) = w, wsupp
@inline compute_v(m::NScalarLinearAdvection, w, wsupp) = w
@inline invert_v(m::NScalarLinearAdvection, v) = v

function compute_w_int(m::NScalarLinearAdvection, w_L, wsupp_L, w_R, wsupp_R)
    return (w_L + w_R)/2, (wsupp_L + wsupp_R)/2
end

function flux(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D}
    return SVector{N, T}(w[1:N] * m.velocity[1])
    # Only the first coordinate in the frame of the interface, i.e. the normal vector
end

@inline rotate_flux(F, m::NScalarLinearAdvection, rotation_matrix) = F
@inline eigenvalues(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D} = @SVector fill(m.velocity[1], N)

@inline left_eigenvectors(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D} = SMatrix{N, N, T}(I)
@inline right_eigenvectors(m::NScalarLinearAdvection{N, T, D}, w, wsupp) where {N, T, D} = SMatrix{N, N, T}(I)
