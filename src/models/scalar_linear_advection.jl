
struct ScalarLinearAdvection{T, D}
    velocity::SVector{D, T}
end

ScalarLinearAdvection(v) = ScalarLinearAdvection{typeof(v), 1}(SVector{1, typeof(v)}(v))

Base.eltype(m::ScalarLinearAdvection{T, D}) where {T, D} = T
nb_dims(m::ScalarLinearAdvection{T, D}) where {T, D} = D
nb_vars(m::ScalarLinearAdvection{T, D}) where {T, D} = 1
nb_vars_supp(m::ScalarLinearAdvection{T, D}) where {T, D} = 0

w_names(m::ScalarLinearAdvection{T, D}) where {T, D} = (:α,)
wsupp_names(m::ScalarLinearAdvection{T, D}) where {T, D} = Tuple([])

@inline rotate_model(m::ScalarLinearAdvection, rotation_matrix) = ScalarLinearAdvection(rotation_matrix * m.velocity)

@inline compute_wsupp(m::ScalarLinearAdvection, w) = SVector{0, Float64}()
@inline rotate_state(w, wsupp, m::ScalarLinearAdvection, rotation_matrix) = w, wsupp
@inline compute_v(m::ScalarLinearAdvection, w, wsupp) = w
@inline invert_v(m::ScalarLinearAdvection, v) = v

function compute_w_int(m::ScalarLinearAdvection, w_L, wsupp_L, w_R, wsupp_R)
    return (w_L + w_R)/2, (wsupp_L + wsupp_R)/2
end

function flux(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D}
    return SVector{1, T}(w[1] * m.velocity[1])
    # Only the first coordinate in the frame of the interface, i.e. the normal vector
end

@inline rotate_flux(F, m::ScalarLinearAdvection, rotation_matrix) = F
@inline eigenvalues(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D} = SVector{1, T}(m.velocity[1])

@inline left_eigenvectors(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D} = SMatrix{1, 1, T}(1.0)
@inline right_eigenvectors(m::ScalarLinearAdvection{T, D}, w, wsupp) where {T, D} = SMatrix{1, 1, T}(1.0)
