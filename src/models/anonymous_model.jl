# Anonymous = only defined by a flux function

struct ScalarAnonymousModel{D} <: AbstractModel
    flux::SVector{D, Function}
end

function ScalarAnonymousModel(flux::AbstractArray)
    D = length(flux)
    ScalarAnonymousModel{D}(SVector{D, Function}(flux...))
end

ScalarAnonymousModel(flux::Function) = ScalarAnonymousModel([flux])

nb_vars(m::ScalarAnonymousModel) = 1
eltype(m::ScalarAnonymousModel) = Float64
nb_dims(m::ScalarAnonymousModel{D}) where D = D

w_names(m::ScalarAnonymousModel) = (:u,)

function rotate_model(m::ScalarAnonymousModel{D}, rotation_matrix, position) where D
    fx = u -> sum(rotation_matrix[i, 1] * m.flux[i](u) for i in 1:D)
    ScalarAnonymousModel{1}([fx])
end

normal_flux(m::ScalarAnonymousModel, w) = m.flux[1](w)

using ForwardDiff
using LinearAlgebra
eigenvalues(m::ScalarAnonymousModel{D}, w::Number) where {D} = eigenvalues(m, SVector{1, typeof(w)}(w))
eigenvalues(m::ScalarAnonymousModel{D}, w::SVector) where {D} = eigvals(ForwardDiff.jacobian(w -> m.flux[1](w), w))

# left_eigenvectors(m::ScalarAnonymousModel{D}, w) where {D} = inv(eigen(ForwardDiff.jacobian(w -> m.flux[1](w), w)).vectors)
# right_eigenvectors(m::ScalarAnonymousModel{D}, w) where {D} = eigen(ForwardDiff.jacobian(w -> m.flux[1](w), w)).vectors

left_eigenvectors(m::ScalarAnonymousModel{D}, w) where {D} = SMatrix{1, 1, Float64}(I)[1]
right_eigenvectors(m::ScalarAnonymousModel{D}, w) where {D} = SMatrix{1, 1, Float64}(I)[1]
