function [g] = sigmoid(z)
%SIGMOID Calculates the sigmoid function with input z
g = 1 ./ (1 + exp(-z));
end

