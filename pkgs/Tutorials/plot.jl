#= Optimize a pulse to find the ground-state energy of a molecular Hamiltonian. =#

##########################################################################################
#= PREAMBLE =#
import Plots
import JLD2: load,save

chemical_accuracy = 0.001593601
dir =  "/Users/gxc/Documents/Projects/9_CtrlVQE_TunableCoupler/data"
mole = "H2"
W = 50
r = 1000
init_g = 0.002
maxiter = 10000
T = 30.0
dist = 0.5

m = 2

Tlist = []

Tmin = 1
Tmax = 10
Tstep = 1

for T in Tmin:Tstep:Tmax
    push!(Tlist, Float64(T))
end

Tmin = 10
Tmax = 50
Tstep = 5

for T in Tmin:Tstep:Tmax
    push!(Tlist, Float64(T))
end

distmin = 1
distmax = 25
diststep = 1

distlist = distmin:diststep:distmax
##########################################################################################
#= DATA DIRECTORY AND FILE NAME=#
devices = ["fixedcoupling", "fixedFrequencyTunableCoupling", "tunableCouplingNoLocalDrives"]
minevol_vs_dist = []

tracearray_device = Dict()

for key in devices
    trace_vs_dist = Dict()
    for j in eachindex(distlist)
        trace = Dict()
        # tracelist = Vector{Dict{String,Any}}(undef, length(Tlist))
        for i in eachindex(Tlist)
            T = Float64(Tlist[i])
            dist = Float64(distlist[j]/10)
            if key == "fixedcoupling"
                fn = "trace.$mole.$key.W.$W.T.$T.r.$r.maxiter.$maxiter.m.$m.dist.$dist.jld2"
            else
                fn = "trace.$mole.$key.W.$W.T.$T.r.$r.initg.$init_g.maxiter.$maxiter.m.$m.dist.$dist.jld2"
            end
            subdir = "$dir/molecular_hamiltonian/$key/$(mole)_$(dist)"
            fn_fp = "$subdir/$fn"

            if isdir(subdir)
                if isfile(fn_fp)
                    trace = merge(trace, Dict(T => load(fn_fp)))
                else
                    println("File not found: W = $W, T = $T, dist = $dist")
                    continue
                end
            end
        end
        trace_vs_dist = merge(trace_vs_dist, Dict(dist => trace))
    end
    tracearray_device = merge(tracearray_device, Dict(key => trace_vs_dist))
    println(key)
end

minevol_vs_dist_vs_device = Dict()
for device in devices
    minevol_vs_dist = []
    for dist in (distlist./10)
        minevol = findfirst([tracearray_device[device][dist][T]["energy error"] for T in Tlist ] .<= chemical_accuracy)
        if isnothing(minevol)
            continue
        end
        append!(minevol_vs_dist, [[dist, Tlist[minevol]]])
    end
    minevol_vs_dist_vs_device = merge(minevol_vs_dist_vs_device , Dict(device => minevol_vs_dist))
end

device="fixedcoupling"
matrix = "pyscf_$(mole)_sto-3g_singlet_$(dist)_P-m"      # MATRIX FILE
H = NPZ.npzread("$(@__DIR__)/matrix/$matrix.npy")
dist = 1.6
T = 30.0
traningsteps_vs_T = []
for T in Tlist
    if tracearray_device[device][dist][T]["energy error"] <= chemical_accuracy
        append!(traningsteps_vs_T,[[T,length(tracearray_device[device][dist][T]["energy"])]])
    end
end

colorlist = [:black, :red, :blue]
plotlegends = Dict("tunableCouplingNoLocalDrives" => "g(t)", "fixedFrequencyTunableCoupling" => "g(t) , Ω(t)", "fixedcoupling" => "Ω(t)")
device = "fixedFrequencyTunableCoupling";
plot=Plots.plot();

colori = 1;
for device in keys(minevol_vs_dist_vs_device)
    xdata = [ vec[1] for vec in minevol_vs_dist_vs_device[device]]
    ydata = [ vec[2] for vec in minevol_vs_dist_vs_device[device]]
    Plots.plot!(plot, xdata, ydata,
        label = plotlegends[device],
        legend = :bottomright,
        framestyle = :box,
        color = colorlist[colori]
    )
    colori += 1
end
display(plot)
for αind in eachindex(αlist)
    α = αlist[αind]
    xdata = 1:length(c_dict[α][T][1])
    ydata = [ count([ c_dict[α][T][seed][iter] > thres for seed in sort(collect(keys(c_dict[α][T])))  ]) / length(sort(collect(keys(c_dict[α][T])))) for iter in 1:length(c_dict[α][T][1]) ]

    Plots.plot!(plot1,xdata,ydata,
    color = colorlist[αind],
    # markershape = :circle,
    linestyle = :dash,
    label = "Ascending cVaR, min(α) = $α"
    )
end



xmax = 10^ceil(log10(length(c_dict[α][T][1])))

Plots.ylims!(plot1,0,1.1)
Plots.xlims!(plot1,0.9,xmax)
Plots.xaxis!(plot1,:log)
Plots.ylabel!(plot1,"Fraction of |⟨ψ(T)|0⟩|² > 0.1")
Plots.xlabel!(plot1,"Optimization step")
Plots.annotate!(plot1,0.7*xmax,0.4,Plots.text("v = $v, d = $(v-1), T = $T, LBFGSB", :black, :right, 10))
display(plot1)

dir = "$(@__DIR__)/dat-QAOA/cVaR-moredata/ascending_alpha/L-BFGS-B_SciPy/fig/v.$v.d.$(v-1)"
if !isdir(dir)
    mkpath(dir)
end

import FileIO: save
save("$dir/FractionvsIteraction-LBFGSB.v.$v.d.$(v-1).W.$W.T.$T.pdf",plot1)