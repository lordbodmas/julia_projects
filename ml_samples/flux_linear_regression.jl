using Flux, Statistics, DataFrames, CSVFiles
using Flux: Params, gradient
using Flux.Optimise: update!
using Parameters: @with_kw

"""
https://github.com/FluxML/model-zoo/blob/master/other/housing/housing.jl
"""

# Struct to define hyperparameters
@with_kw mutable struct Hyperparams
    lr::Float64 = 0.1		# learning rate
    split_ratio::Float64 = 0.1	# Train Test split ratio, define percentage of data to be used as Test data
end

function load_data(fpath, args)
    df = DataFrame(load(CSV_FPATH; header_exists=true))


end
