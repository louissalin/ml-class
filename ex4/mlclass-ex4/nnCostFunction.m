function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];

for i = 1:m
	%% FP
	x = X(i, :);
	z2 = Theta1 * x';
	a2 = sigmoid(z2);
	a2 = [1; a2];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	%% Cost
	y_vec = zeros(num_labels, 1);
	y_vec(y(i)) = 1;

	for k = 1:num_labels
		cost = y_vec(k) * log(a3(k)) + (1 - y_vec(k)) * log(1-a3(k));
		J = J + cost;
	end
	
	%% BP
	d3 = a3 - y_vec;

	%%% -- NOTE --
	%%% z2 is missing the bias that's added to a2 above, so add it below
	%%% otherwise the vector sizes will be off by 1
	d2 = (Theta2' * d3) .* [1; sigmoidGradient(z2)];

	Theta2_grad = Theta2_grad + d3 * a2';

	%%% we must now remove the extra number we added when computing d2 above
	Theta1_grad = Theta1_grad + d2(2:end) * x;
end

%% Gradients
Theta2_grad = Theta2_grad ./ m;
Theta1_grad = Theta1_grad ./ m;

J = -J / m;

%% regularize J
sum_theta1 = 0;
for i = 1:size(Theta1, 1)
	for j = 2:size(Theta1, 2)
		sum_theta1 = sum_theta1 + (Theta1(i,j)^2);
	end
end

sum_theta2 = 0;
for i = 1:size(Theta2, 1)
	for j = 2:size(Theta2, 2)
		sum_theta2 = sum_theta2 + (Theta2(i,j)^2);
	end
end

J = J + (lambda / (2 * m)) * (sum_theta1 + sum_theta2);

%% regularize gradients
for i = 1:size(Theta1_grad, 1)
	for j = 2:size(Theta1_grad, 2)
		Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m * Theta1(i,j));
	end
end

for i = 1:size(Theta2_grad, 1)
	for j = 2:size(Theta2_grad, 2)
		Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m * Theta2(i,j));
	end
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
