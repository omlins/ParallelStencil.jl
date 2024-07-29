# Activate the current environment and load all packages
using Pkg
Pkg.activate(@__DIR__)

using DelimitedFiles, Statistics

function prepare_data()

     regex = Regex("out_diffusion3D_clean_benchm.")

     ne = length(readdir(joinpath(@__DIR__, "../data")))
     data = Array{Union{Float64,String},2}(undef, ne, 4)

     i = 1
     for fl in readdir(joinpath(@__DIR__, "../data"))
          if splitext(fl)[end] != ".txt"
               continue
          end
          data[i, 1] = replace(splitext(fl)[1], regex => "")
          data[i, 2] = median(readdlm(joinpath(@__DIR__, "../data", fl))[:, end])
          # 95% of confidence interval
          tmp = sort(readdlm(joinpath(@__DIR__, "../data", fl))[:, end]; rev=true)
          data[i, 3] = tmp[5]  # 5th rank  <=   (n-1.96*n^(1/2))/2 =  5.617306764100412
          data[i, 4] = tmp[16] # 16th rank <= 1+(n+1.96*n^(1/2))/2 = 15.382693235899588
          # previous naive approach
          # data[i,3] = minimum(readdlm(joinpath(@__DIR__,"../data",fl))[:,end])
          # data[i,4] = maximum(readdlm(joinpath(@__DIR__,"../data",fl))[:,end])
          i += 1
     end
     return data
end

data = prepare_data()

TpeakA100 = 1370
TpeakP100 = 559
TpeakEPYC = 170 # vendor announced (2x85)
TpeakXeon = 57

using StatsPlots, LaTeXStrings, Plots.Measures

default(fontfamily="Computer Modern", margin=5mm); scalefontsizes(); scalefontsizes(1.1)

# NOTE: I only used the BenchmarkTools generated results here
exp_gpu = [3, 2, 1]#["broadcasting", "explicit", "math-close"]
nam_gpu = repeat(exp_gpu, outer=2)
sx_gpu  = repeat(["P100", "A100"], inner=length(exp_gpu))
id_A100 = [22, 16, 2]
id_P100 = [21, 15, 1]
data_gpu     = [Float64.(data[id_P100, 2]); Float64.(data[id_A100, 2])]
data_gpu_min = [Float64.(data[id_P100, 3]); Float64.(data[id_A100, 3])]
data_gpu_max = [Float64.(data[id_P100, 4]); Float64.(data[id_A100, 4])]
data_gpu_σ   = data_gpu_max .- data_gpu_min

exp_cpu  = [3, 2, 1]#["broadcasting", "explicit", "math-close"]
nam_cpu  = repeat(exp_cpu, outer=2)
sx_cpu   = repeat(["Xeon", "EPYC"], inner=length(exp_cpu))
id_EPYC  = [10, 18, 6]
id_Xeon  = [9, 17, 5]
data_cpu     = [Float64.(data[id_Xeon, 2]); Float64.(data[id_EPYC, 2])]
data_cpu_min = [Float64.(data[id_Xeon, 3]); Float64.(data[id_EPYC, 3])]
data_cpu_max = [Float64.(data[id_Xeon, 4]); Float64.(data[id_EPYC, 4])]
data_cpu_σ   = data_cpu_max .- data_cpu_min

p1 = plot([0.5, 3.0], [TpeakA100, TpeakA100], lw=2, lc=:green, label=false)
plot!([0.5, 3.5], [TpeakP100, TpeakP100], lw=2, lc=:lightgreen, label=false)
groupedbar!(nam_gpu, data_gpu, group=sx_gpu, yerr=data_gpu_σ,
     bar_width=0.6, c=[:green :lightgreen],
     # xaxis=nothing,
     xticks=(exp_gpu, ["broadcasting", "explicit", "math-close"]),
     yticks=([250, 500, 750, 1000, 1250], ["250 (10.4)", "500 (20.8)", "750 (31.2)", "1000 (41.7)", "1250 (52.1)"]),
     ylims=(0, 1490),
     framestyle=:box, grid=false, xrotation=60,
     ylabel=L"T_\mathrm{eff}" * " [GB/s]" * "  (Gpts/s)",
     foreground_color_legend=nothing,
     right_margin=1mm,
)
annotate!(p1, .18, 0, text(L"0", :left, 10)) # workaround as yticks can only handle 5 items...
annotate!(p1, 0.5, TpeakA100 * 1.04, text(L"T_\mathrm{peak}= " * string(TpeakA100) * " [GB/s]", :green, :left, 6))
annotate!(p1, 3.5, TpeakP100 * 1.08, text(L"T_\mathrm{peak}= " * string(TpeakP100) * " [GB/s]", :green, :right, 6))

# p2=plot([0.5, 3.5],[TpeakXeon,TpeakXeon], lw = 2, lc = :lightblue, label = false)
# plot!([0.5, 3.5],[TpeakEPYC,TpeakEPYC], lw = 2, lc = :blue, label = false)
p2 = groupedbar(nam_cpu, data_cpu, group=sx_cpu, yerr=data_cpu_σ,
     bar_width=0.6, c=[:blue :lightblue],
     xticks=(exp_cpu, ["broadcasting", "explicit", "math-close"]),
     ymirror=true,
     yticks=([10, 20, 30, 40, 50], ["10 (0.42)", "20 (0.83)", "30 (1.2)", "40 (1.7)", "50 (2.1)"]),
     ylims=(0, 65),
     framestyle=:box, grid=false, xrotation=60,
     foreground_color_legend=nothing,
     left_margin=1mm,
)
annotate!(p2, 3.5, 0, text(L"\;0", :left, 10)) # workaround as yticks can only handle 5 items...
annotate!(p2, 3.5, 60, text(L"60 \;\; (2.5)", :left, 10)) # workaround as yticks can only handle 5 items...

display(plot(p1, p2, dpi=150))

png("julia_xpu_Teff.png")
