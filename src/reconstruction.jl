
minmod      = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? min(a, b) : max(a, b))
superbee    = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? max(min(2*a, b), min(a, 2*b)) : min(max(2*a, b), max(a, 2*b)))
ultrabee(β) = (a, b) -> a*b <= 0 ? 0.0 : (a >= 0 ? 2*max(0, min((1/β-1)*a, b)) : -ultrabee(β)(-a, -b))

all_cells(wi) = true
no_cell(wi) = false

identity(x) = x

function muscl_reconstruction(limiter, flag=all_cells, renormalize=identity)
	function reconstruction(grid, model, w, wsupp, i_cell, i_face)
		if flag(w[i_cell])
			left_∇w = left_gradient(grid, w, i_cell)
			right_∇w = right_gradient(grid, w, i_cell)
			dx = face_center_relative_to_cell(grid, i_cell, i_face)' * rotation_matrix(grid, i_face)[:, 1]
			if dx > 0.0
				limited_∇w = limiter.(left_∇w, right_∇w)
			else
				limited_∇w = limiter.(right_∇w, left_∇w)
			end
			reconstructed_w = w[i_cell] + dx*limited_∇w
			reconstructed_w = renormalize(reconstructed_w)
			reconstructed_wsupp = compute_wsupp(model, reconstructed_w)
			return rotate_state(reconstructed_w, reconstructed_wsupp, model, rotation_matrix(grid, i_face))
		else
			return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
		end
	end
end

muscl(params...) = (args...) -> in_local_coordinates(local_upwind_flux, args...; reconstruction=muscl_reconstruction(params...))

function vof_reconstruction(method, flag=all_cells, i_field=1)
    function reconstruction(grid, model, w, wsupp, i_cell, i_face)
        if flag(w[i_cell])
            st = oriented_stencil(grid, i_cell, i_face)
            α = get_component(w[st], i_field)
            #= α_flux = method(α, β) =#
            α_flux = method(α)
            return rotate_state([α_flux], wsupp[i_cell], model, rotation_matrix(grid, i_face))
        else
            return rotate_state(w[i_cell], wsupp[i_cell], model, rotation_matrix(grid, i_face))
        end
    end
end

vof_flux(params...) = (args...) -> in_local_coordinates(local_upwind_flux, args...; reconstruction=vof_reconstruction(params...))

get_component(w::AbstractArray, i) = (x -> x[i]).(w)

function upwind_stencil(grid,
                        model::ScalarLinearAdvection,
                        w, wsupp, i_face)
    i_cell_1, i_cell_2 = cells_next_to_inner_face(grid, i_face)
    w₁, wsupp₁ = rotate_state(w[i_cell_1], wsupp[i_cell_1], model, rotation_matrix(grid, i_face))
    velocity = model.velocity[1]
    upwind_cell = velocity > 0.0 ? i_cell_1 : i_cell_2
    st = oriented_stencil(grid, upwind_cell, i_face)[0, :]
    return velocity, w[st]
end

function stability_range(α, β)
    maxi = max(α[-1], α[0])
    mini = min(α[-1], α[0])
    bornesup = min((α[0] - mini)/β + mini, max(α[0], α[1]))
    borneinf = max((α[0] - maxi)/β + maxi, min(α[0], α[1]))
    return (borneinf, bornesup)
end

cut_in_range(inf, sup, x) = min(sup, max(inf, x))

const β = 0.2

function lagoutiere_downwind_flux(grid::FaceSplittedMesh{PeriodicRegularMesh2D},
                                  model::ScalarLinearAdvection{1, T, D},
                                  w, wsupp, i_face) where {T, D}
    velocity, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    wst = get_component(wst, 1)
    borneinf, bornesup = stability_range(wst, β)
    α_flux = cut_in_range(borneinf, bornesup, wst[1])
    ϕ = eltype(w)(velocity*α_flux)
    return ϕ
end

function lagoutiere_downwind_flux(grid::FaceSplittedMesh{PeriodicRegularMesh2D},
                                  model::ScalarLinearAdvection{N, T, D},
                                  w, wsupp, i_face) where {N, T, D}
    velocity, wst = upwind_stencil(grid, model, w, wsupp, i_face)
    bornesup = zeros(nb_vars(model))
    borneinf = zeros(nb_vars(model))
    for i in 1:nb_vars(model)
        wi = get_component(wst, i)
        borneinf[i], bornesup[i] = stability_range(wi, β)
    end
    for i in 1:nb_vars(model)
        updated_borneinf = max(borneinf[i], 1.0 - sum(α_flux[j] for j in 1:(i-1)) - sum(bornesup[j] for j in (i+1):nb_vars(model)))
        updated_bornesup = min(bornesup[i], 1.0 - sum(α_flux[j] for j in 1:(i-1)) - sum(borneinf[j] for j in (i+1):nb_vars(model)))
        α_flux[i] = min(updated_bornesup, max(updated_borneinf, wi[1]))
    end

    ϕ = velocity*α_flux
    return ϕ
end


