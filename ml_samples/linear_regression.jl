"""
Julia implementation of linear regression from scratch.

Code taken from https://towardsdatascience.com/julia-for-data-science-how-to-build-linear-regression-from-scratch-with-julia-6d1521a00611

## Data
Rather than the data in the tutorial, we used some CDC data about obesity trends in the US.
https://www.kaggle.com/spittman1248/cdc-data-nutrition-physical-activity-obesity

"""
# Install of the packages. You'll need to register an account with Julia
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Statistics")

using DataFrames
using CSVFiles

CSV_FPATH = "D:/julia_projects/data/CDC/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv"

df = DataFrame(load(CSV_FPATH; header_exists=true))

println(size(df))

# hypothesis hθ(x) = θ₀ + θ₁x
hθ(x, θ) = θ[1] + θ[2]*x
