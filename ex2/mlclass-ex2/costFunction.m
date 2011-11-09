function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

g = sigmoid(X * theta);
costs = (log(g).* -y) - (log(1 .- g) .* (1 .- y));

J = sum(costs);
J = J / m;

grad = (X' * (g - y)) ./ m;

%g = sigmoid(X * theta);
%for i=1:m
	%cost = -y(i)*log(g(i)) - (1-y(i))*log(1-g(i));
	%J = J + cost;
%end

%J = J / m;

%for j=1:size(theta, 1)
	%for i=1:m
		%grad(j) = grad(j) + (g(i) - y(i))*X(i,j);
	%end

	%grad(j) = grad(j) / m;
%end


% =============================================================

end
