using Statistics
using Flux.Data: Iris
using Random: shuffle!
using Plots

"""
Code is modified from https://towardsdatascience.com/julia-for-data-science-regularized-logistic-regression-667857a7f0ce
"""

# unzip vector of tuples
unzip(a) = map(x -> getfield.(a, x), fieldnames(eltype(a)))


normalise_features(x) = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

# sigmoid activation function
sigmoid_activation(z) = 1 ./ (1 .+ exp.(.-z))


function split_data(x, y, split_ratio)
    split_index = round(Int, length(y)*split_ratio)
    x_train = x[1:split_index, :]
    y_train = y[1:split_index, :]
    x_test = x[split_index+1:size(x,1), :]
    y_test = y[split_index+1:size(x,1), :]
    train_data = (x_train, y_train)
    test_data = (x_test, y_test)
    return train_data, test_data
end


function test_split_data()
    X = [1 2 3; 2 4 6; 7 8 9]
    y = [0, 0, 1]
    (X_train, y_train), (X_test, y_test) = split_data(X, y, 0.66)
    println((X_train, y_train))
    println((X_test, y_test))
end


function do_shuffle_data(features, labels)
    zipped = collect(zip(features, labels))
    shuffle!(zipped)
    features, labels = unzip(zipped)
    return features, labels
end


function load_iris_data(; shuffle_data=false)
    """
    Loads the Iris classification data set via flux. Splits into train and test
    examples
    """
    features = Iris.features()
    normed_features = normalise_features(features)
    normed_features = permutedims(normed_features) # columns and rows need to be swapped?
    labels = Iris.labels()

    # for simplicity, convert to a binary classification problem
    unique_sorted_labels = sort(unique(labels))
    target_label = unique_sorted_labels[1]
    binarised_labels = []
    for label in labels
        if label == target_label
            append!(binarised_labels, 1)
        else
            append!(binarised_labels, 0)
        end
    end

    if shuffle_data
        normed_features, lables = do_shuffle_data(normed_features, binarised_labels)
    end

    train_data, test_data = split_data(normed_features, binarised_labels, 0.4)# args.split_ratio)


    return train_data, test_data
end


function cost(X, y, θ)
    """
    Calculate the cost function for features X, labels y given some the weight vector θ
    """
    m = length(y)

    # Sigmoid predictions at current batch
    h = sigmoid_activation(X * θ)

    # Cost(hθ(x), y) = -ylog(hθ(x)) - (1 - y)log(hθ(x))
    # left side of the cost function
    positive_class_cost = ((-y)' * log.(h))

    # right side of the cost function
    negative_class_cost = ((1 .- y)' * log.(1 .- h))

    # \bf for bolded J
    𝐉 = (1/m) * (positive_class_cost - negative_class_cost)
    # X' is adjugate matrix
    ∇𝐉 = (1/m) * (X') * (h-y)
    return (𝐉, ∇𝐉)
end


function logistic_regression_sgd(X, y; fit_intercept=true, η=0.01, max_iter=1000)
    """
    This function uses gradient descent to search for the weights
    that minimises the logit cost function.
    A tuple with learned weights vector (θ) and the cost vector (𝐉)
    are returned.
    """

    # Initialize some useful values
    m = length(y); # number of training examples
    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X # Assume user added constants
    end
    # Use the number of features to initialise the theta θ vector
    n = size(X)[2]
    θ = zeros(n)

    # Initialise the cost vector based on the number of iterations
    𝐉 = [] #zeros(max_iter)

    for iter in range(1, stop=max_iter)
        # Calcaluate the cost and gradient (∇𝐉) for each iter
        #𝐉[iter], ∇𝐉 = cost(X, y, θ)
        𝐉_iter, ∇𝐉 = cost(X, y, θ)
        append!(𝐉, 𝐉_iter)
        # Update θ using gradients (∇𝐉) for direction and (η) for the magnitude of steps in that direction
        θ = θ - (η * ∇𝐉)
    end

    return (θ, 𝐉)
end


function predict_proba(X, θ; fit_intercept=true)
    """
    This function uses the learned weights (θ) to make new predictions.
    Predicted probabilities are returned.
    """
    m = size(X)[1]
    if fit_intercept
        # Add a constant of 1s if fit_intercept is specified
        constant = ones(m, 1)
        X = hcat(constant, X)
    else
        X
    end

    h = sigmoid_activation(X * θ)
    return h
end

function test_predict_proba()
    X = [1 2 3; 2 4 6; 7 8 9]
    θ = [0.05352619366289196, 0.017324481133459118, -0.018877231395973748]
    h = predict_proba(X, θ, fit_intercept=false)
    print(h)
end


function test_logistic_regression_sgd()
    X = [1 2 3; 2 4 6; 7 8 9]
    y = [0, 0, 1]
    θ, 𝐉 = logistic_regression_sgd(X, y, fit_intercept=false, max_iter=10)
    println("printing θ and 𝐉")
    println("theta ", θ)
    println("J ", 𝐉)
end


function test_cost()
    X = [1 2 3; 4 5 6; 7 8 9]
    y = [1, 2, 1]
    θ = [1, 1, 1]
    @assert cost(X, y, θ) == (-4.999174669566721, [-1.334157949009934, -1.6683155920421386, -2.0024732350743433])
end


function predict_class(proba, threshold=0.5)
    """
    This function binarizes predicted probabilities using a threshold.
    Default threshold is set to 0.5
    """
    return proba .>= threshold
end


function train_and_evaluate()
    (X_train, y_train), (X_test, y_test) = load_iris_data()
    θ, 𝐉 = logistic_regression_sgd(X_train, y_train)
    train_predictions = predict_class(predict_proba(X_train, θ))
    train_score = mean(y_train .== train_predictions);
    test_score = mean(y_test .== predict_class(predict_proba(X_test, θ)));

    println("Training score: ", round(train_score, sigdigits=4))

    println("Testing score: ", round(test_score, sigdigits=4))
    return 𝐉

end


𝐉 = train_and_evaluate()
# Plot the cost vector
plot(𝐉, color="blue", title="Cost Per Iteration", legend=false,
     xlabel="Num of iterations", ylabel="Cost")
