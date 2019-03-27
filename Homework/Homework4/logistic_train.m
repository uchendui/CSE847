function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix with n samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
if ~exist('epsilon','var')
    epsilon=1e-5;
end

if ~exist('maxiter','var')
    maxiter=1000;
end


% Initialize weight vector to all zeros
w = zeros(size(data,2),1);
size(w);
lr = 0.001;
prev_predictions = zeros(size(data,1),1);
for t = 0:maxiter
    % Early stopping conditions
    predictions = sigmoid( data * w); 
    if(mean(abs(predictions - prev_predictions)) < epsilon)
        break;
    end
    prev_predictions = predictions;
    
    % Compute the gradient
    s = sigmoid(-labels .* (data * w));
    g = labels .* (data .* s);
    g = mean(g)';
    
    % Update the weights (along the negative gradient):
    w = w + lr * g;
end

weights = w;
end
