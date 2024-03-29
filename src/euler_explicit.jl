# A simple Euler explicit time stepping method to use and test the other
# methods of the package without requiring a full ODE solver.

import ProgressMeter

struct FixedCourant{T}
    fixed_courant::T
end

struct TimeDependantFlux{F}
    flux_at::F
end

function directional_splitting(f::TimeDependantFlux)
    (
    TimeDependantFlux(t -> directional_splitting(f.flux_at(t))[1]),
    TimeDependantFlux(t -> directional_splitting(f.flux_at(t))[2]),
    )
end


function run!(fluxes::Tuple, mesh, w, scheme, t; nb_time_steps, time_step, verbose=true, callback=nothing, kwargs...)

	if verbose
		p = ProgressMeter.Progress(nb_time_steps, dt=0.1)
	end

    current_fluxes = map(f -> f isa TimeDependantFlux ? f.flux_at(t) : f, fluxes)
    Δw = zeros(Δw_type(current_fluxes[1], mesh, w, scheme, t), size(w))

    for i_time_step in 1:nb_time_steps

        current_fluxes = map(f -> f isa TimeDependantFlux ? f.flux_at(t) : f, fluxes)

        if time_step isa FixedCourant
            dt = minimum(time_step.fixed_courant/courant(one(eltype(t)), f, mesh, w) for f in current_fluxes)
        else
            dt = time_step
        end

        for f in current_fluxes
            Δw .= zeros(eltype(Δw), size(Δw))
            div!(Δw, f, mesh, w, scheme, dt)
            w .-= dt .* Δw
        end

		if verbose
			ProgressMeter.update!(p, i_time_step)
		end

        t += dt

        if callback != nothing
            callback(i_time_step, t, w)
        end
    end

    return t
end

run!(flux, mesh, w, scheme, t; kwargs...) = run!((flux,), mesh, w, scheme, t; kwargs...)

run!(flux, mesh, w, scheme; t=0.0, kwargs...) = run!(flux, mesh, w, scheme, t; kwargs...)

run!(flux, mesh, w; scheme=Upwind(), boundary_flux=NeumannBC(), kwargs...) = run!(flux, mesh, w, (scheme, boundary_flux); kwargs...)

function run(flux, mesh, w₀, args...; kwargs...)
	w = deepcopy(w₀)
    t = run!(flux, mesh, w, args...; kwargs...)
	return t, w
end

