
struct ShallowWater{D, T} <: AbstractModel
    g::T
end


Base.eltype(m::ShallowWater{D, T}) where {D, T} = T
nb_dims(m::ShallowWater{D, T}) where {D, T} = D
nb_vars(m::ShallowWater{D, T}) where {D, T} = 1 + D
consvartype(m::ShallowWater{D, T}, w) where {D, T} = SVector{1 + D, T}

########################################

########
#  1D  #
########

w_names(m::ShallowWater{1, T}) where T = (:h, :u)

function rotate_state(w, m::ShallowWater{1, T}, rotation_matrix) where T
    return (h=w.h, u=w.u * rotation_matrix[1, 1])
end

function compute_v(m::ShallowWater{1, T}, w::NamedTuple) where T
    h, u = w.h, w.u
    return SVector{2, eltype(m)}(h, h*u)
end

invert_v(m::ShallowWater{1, T}, v) where T = (h=v[1], u=v[2]/v[1])

function normal_flux(m::ShallowWater{1, T}, w::NamedTuple) where T
    h, u = w.h, w.u
    return SVector(h*u, h*u^2 + h^2*m.g/2)
end

compute_v(m::ShallowWater, w::AbstractArray) = compute_v(m, (h=w[1], u=w[2]))
normal_flux(m::ShallowWater, w::AbstractArray) = normal_flux(m, (h=w[1], u=w[2]))

function rotate_flux(F, m::ShallowWater{1, T}, rotation_matrix) where T
    SVector{2, T}(F[1], F[2] * rotation_matrix[1, 1])
end

