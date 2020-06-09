s1 = "The quick brown fox jumps over the lazy dog α,β,γ"
println(s1)


# Int('1') gives you the ascii value of a char
println(Int('1'))

s1_caps = uppercase(s1)
println(s1_caps)
s1_lowercase = lowercase(s1)


# substrings
show(s1[1:10]); println()

# end is used for the end of the array or string
show(s1[end-10: end]); println()


# shortcut for characters like α is to press backspace
α = 0.8
println(α)

# string interpolation
a = "welcome"
b = "julia"

println("$a to $b")

# this can be extended to evaluate statements
println("1 + 2 = $(1 + 2)")

# strings are concatenated using the * operator
s2 = "this" * " and" * " that"
println(s2)
# as well as the string function
s3 = string("this", " and", " that")
println(s3)
