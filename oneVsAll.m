function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

    for c = 1:num_labels 
        all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
        % remember y (5000*1) is an array of labels i.e. it contains actual 
        % digit names (y==c) will return a vector with values 0 or 1. 1 at places where y==c 
        
        % 't' is passed as dummy parameter which is initialized with 'initial_theta' first
        % then subsequent values are choosen by fmincg [Note: Its not a builtin function like fminunc
        
        % fmincg will consider all training data having label c (1-10 note
        % 0 is mapped to 10) and find the optimal theta vector for it (Classifying white pixels with gray pixels). same
        % process is repeated for other classes
    end
end
