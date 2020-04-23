all_cells(wi) = true
no_cell(wi) = false

identity(x) = x


###########
#  Muscl  #
###########

Base.@kwdef struct Muscl
    limiter
    flag=all_cells
    renormalize=identity
end

minmod      = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? min(a, b) : max(a, b))
superbee    = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? max(min(2*a, b), min(a, 2*b)) : min(max(2*a, b), max(a, 2*b)))
ultrabee(β) = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? 2*max(0, min((1/β-1)*a, b)) : -ultrabee(β)(-a, -b))

function (s::Muscl)(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    normal_velocity = model.velocity[1]
    upwind_cell = normal_velocity > 0 ? i_cell_1 : i_cell_2
    if s.flag(w[upwind_cell])
        left_∇w = left_gradient(grid, w, upwind_cell)
        right_∇w = right_gradient(grid, w, upwind_cell)
        dx = face_center_relative_to_cell(grid, upwind_cell, i_face)' * rotation_matrix(grid, i_face)[:, 1]
        if dx > 0.0
            limited_∇w = s.limiter.(left_∇w, right_∇w)
        else
            limited_∇w = s.limiter.(right_∇w, left_∇w)
        end
        reconstructed_w = w[upwind_cell] + dx*limited_∇w
        reconstructed_w = s.renormalize(reconstructed_w)
        return eltype(w)(normal_velocity * reconstructed_w)
    else
        return eltype(w)(normal_velocity * w[upwind_cell])
    end
end

#########
#  VOF  #
#########

Base.@kwdef struct VOF
    method
    flag=all_cells
    renormalize=identity
end

function upwind_stencil(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    local_model = rotate_model(model, rotation_matrix(grid, i_face))
    velocity = local_model.velocity[1]
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    upwind_cell = velocity > 0.0 ? i_cell_1 : i_cell_2
    st = oriented_stencil(grid, upwind_cell, i_face)
    return velocity, w[st]
end

function (s::VOF)(grid, model::ScalarLinearAdvection, w, wsupp, i_face)
    v, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    α_flux = s.flag(w[0, 0]) ? method(α) : w[0, 0]
    return eltype(w)(v * w[upwind_cell])
end


########################
#  LagoutiereDownwind  #
########################

Base.@kwdef struct LagoutiereDownwind
    β::Float64
end

function stability_range(α, β)
    maxi = max(α[-1], α[0])
    mini = min(α[-1], α[0])
    bornesup = min((α[0] - mini)/β + mini, max(α[0], α[1]))
    borneinf = max((α[0] - maxi)/β + maxi, min(α[0], α[1]))
    return (borneinf, bornesup)
end

cut_in_range(inf, sup, x) = min(sup, max(inf, x))

function (s::LagoutiereDownwind)(grid, model::ScalarLinearAdvection{1, T, D}, w, wsupp, i_face) where {T, D}
    if grid isa RegularMesh1D
        v, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    else
        v, wst2d = upwind_stencil(grid, model, w, wsupp, i_face)
        wst = wst2d[0, :]
    end
    borneinf, bornesup = stability_range(wst, s.β)
    α_flux = cut_in_range(borneinf, bornesup, wst[1])
    return eltype(w)(v * α_flux)
end

#= function lagoutiere_downwind_flux(grid::FaceSplittedMesh{PeriodicRegularMesh2D}, =#
#=                                   model::ScalarLinearAdvection{N, T, D}, =#
#=                                   w, wsupp, i_face) where {N, T, D} =#
#=     velocity, wst = upwind_stencil(grid, model, w, wsupp, i_face)[0, :] =#
#=     bornesup = zeros(nb_vars(model)) =#
#=     borneinf = zeros(nb_vars(model)) =#
#=     for i in 1:nb_vars(model) =#
#=         wi = get_component(wst, i) =#
#=         borneinf[i], bornesup[i] = stability_range(wi, β) =#
#=     end =#
#=     for i in 1:nb_vars(model) =#
#=         updated_borneinf = max(borneinf[i], 1.0 - sum(α_flux[j] for j in 1:(i-1)) - sum(bornesup[j] for j in (i+1):nb_vars(model))) =#
#=         updated_bornesup = min(bornesup[i], 1.0 - sum(α_flux[j] for j in 1:(i-1)) - sum(borneinf[j] for j in (i+1):nb_vars(model))) =#
#=         α_flux[i] = min(updated_bornesup, max(updated_borneinf, wi[1])) =#
#=     end =#
#=     ϕ = velocity*α_flux =#
#=     return ϕ =#
#= end =#


