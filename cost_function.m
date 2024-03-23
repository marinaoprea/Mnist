function [J, grad] = cost_function(params, X, y, lambda, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)

  % params -> vector containing the weights from the two matrices
  %           Theta1 and Theta2 in an unrolled form (as a column vector)
  % X -> the feature matrix containing the training examples
  % y -> a vector containing the labels (from 1 to 10) for each
  %      training example
  % lambda -> the regularization constant/parameter
  % [input|hidden|output]_layer_size -> the sizes of the three layers
  
  % J -> the cost function for the current parameters
  % grad -> a column vector with the same length as params
  % These will be used for optimization using fmincg
  
  Theta1 = reshape(params(1 : hidden_layer_size * (input_layer_size + 1)), 
                    hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(params(hidden_layer_size * (input_layer_size + 1) + 1 : length(params)),
                   output_layer_size, hidden_layer_size + 1);
  
  Delta1 = zeros(hidden_layer_size, input_layer_size + 1);
  Delta2 = zeros(output_layer_size, hidden_layer_size + 1);
  
  [m, n] = size(X);
  J = 0;

  col = ones(m, 1);
  X = [col X];
  X = X';
  Z = Theta1 * X;
  A2 = 1 ./ (1 .+ exp(-Z));
  col = ones(1, m);
  A2 = [col; A2];
  Z2 = Theta2 * A2;
  A3 = 1 ./ (1 .+ exp(-Z2));
  
  result = zeros(output_layer_size, m);
  for i = 1:m
    result(y(i), i) = 1;
  endfor
  
  J += -sum(sum(result .* log(A3))) - sum(sum((1 .- result) .* log(1 .- A3)));
  delta = A3 - result;
  Delta2 = Delta2 + delta * A2';
    
  delta2 = Theta2' * delta;
  delta2 = delta2 .* (A2 .* (1 .- A2));
    
  delta2 = delta2 (2 : rows(delta2), 1:columns(delta2));
  Delta1 = Delta1 + delta2 * X';
  
  Delta1 /= m;
  Delta1 (1:rows(Delta1), 2:columns(Delta1)) += lambda * (Theta1(1:rows(Theta1), 2:columns(Theta1)) / m);
  Delta2 /= m;
  Delta2 (1:rows(Delta2), 2:columns(Delta2)) += lambda * (Theta2(1:rows(Theta2), 2:columns(Theta2)) / m);
  
  grad = reshape(Delta1, hidden_layer_size * (input_layer_size + 1), 1);
  grad = [grad; reshape(Delta2, output_layer_size * (hidden_layer_size + 1), 1)];
  J = J / m;

  aux1 = Theta1(1 : rows(Theta1), 2 : columns(Theta1)) .* Theta1(1 : rows(Theta1), 2 : columns(Theta1));
  aux2 = Theta2(1 : rows(Theta2), 2 : columns(Theta2)) .* Theta2(1 : rows(Theta2), 2 : columns(Theta2));
  J = J + lambda * (sum(sum(aux1)) + sum(sum(aux2))) / (2 * m);
endfunction
