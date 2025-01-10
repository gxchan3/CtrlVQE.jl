#= Optimize a pulse to find the ground-state energy of a molecular Hamiltonian. =#

##########################################################################################
#= PREAMBLE =#
import Plots
import JLD2: load,save
dir =  "/Users/gxc/Documents/Projects/9_CtrlVQE_TunableCoupler/data"
mole = "H2"
W = 50
r = 1000
init_g = 0.002
maxiter = 10000
T = 30.0
dist = 0.5

m = 2

Tmin = 10
Tmax = 100
Tstep = 5

Tlist = Tmin:Tstep:Tmax

distmin = 1
distmax = 25
diststep = 1

distlist = distmin:diststep:distmax
##########################################################################################
#= DATA DIRECTORY AND FILE NAME=#
tracearray = Vector{Vector{Dict{String,Any}}}(undef, length(distlist))

for j in 1:length(distlist)
    tracelist = Vector{Dict{String,Any}}(undef, length(Tlist))
    for i in 1:length(Tlist)
        T = Float64(Tlist[i])
        dist = Float64(distlist[j]/10)
        fn = "trace.$mole.fixedcoupling.W.$W.T.$T.r.$r.maxiter.$maxiter.m.$m.dist.$dist.jld2"
        subdir = "$dir/molecular_hamiltonian/fixedcoupling/$(mole)_$(dist)"
        fn_fp = "$subdir/$fn"

        if isdir(subdir)
            if isfile(fn_fp)
                tracelist[i] = load(fn_fp)
            else
                println("File not found: $fn_fp")
            end
        end
    end
    tracearray[j] = tracelist
end