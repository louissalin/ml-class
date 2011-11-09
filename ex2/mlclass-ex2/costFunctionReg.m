function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

g = sigmoid(X * theta);
costs = (log(g).* -y) - (log(1 .- g) .* (1 .- y)) + lambda / (2 * m) * sum(theta(2:end) .^ 2);

J = sum(costs);
J = J / m;

grad = (X' * (g - y)) ./ m;
grad(2:end) = grad(2:end) + lambda / m * theta(2:end);

%g = sigmoid(X * theta);
%for i=1:m
	%cost = -y(i)*log(g(i)) - (1-y(i))*log(1-g(i));
	%J = J + cost;
%end

%J = J / m;

%reg_sum = 0;
%for i=2:size(theta, 1)
	%reg_sum = reg_sum + theta(i)^2;
%end

%J = J + (lambda * reg_sum / (2 * m));

%for j=1:size(theta, 1)
	%for i=1:m
		%grad(j) = grad(j) + (g(i) - y(i))*X(i,j);
	%end

	%if j >= 2,
		%grad(j) = grad(j) + (lambda * theta(j));
	%end

	%grad(j) = grad(j) / m;
%end


% =============================================================

end
