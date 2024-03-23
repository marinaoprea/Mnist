function [classes] = predict_classes(X, weights, ...
                  input_layer_size, hidden_layer_size, ...
                  output_layer_size)
  % X -> the test examples for which the classes must be predicted
  % weights -> the trained weights (after optimization)
  % [input|hidden|output]_layer_size -> the sizes of the three layers
  
  % classes -> a vector with labels from 1 to 10 corresponding to
  %            the test examples given as parameter
  
  Theta1 = reshape(weights(1 : hidden_layer_size * (input_layer_size + 1)), 
                   hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(weights(hidden_layer_size * (input_layer_size + 1) + 1 : length(weights)),
                   output_layer_size, hidden_layer_size + 1);

  [m, n] = size(X);
  classes = zeros(m, 1);
  
  col = ones(m, 1);
  X = [col X];
  X = X';
  Z = Theta1 * X;
  A2 = 1 ./ (1 .+ exp(-Z));
  col = ones(1, m);
  A2 = [col; A2];
  Z2 = Theta2 * A2;
  A3 = 1 ./ (1 .+ exp(-Z2));
  for i = 1 : m
    [maxim, classes(i)] = max(A3(:, i));
  endfor
  
endfunction
