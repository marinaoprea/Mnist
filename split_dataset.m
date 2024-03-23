function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  % X -> the loaded dataset with all training examples
  % y -> the corresponding labels
  % percent -> fraction of training examples to be put in training dataset
  
  % X_[train|test] -> the datasets for training and test respectively
  % y_[train|test] -> the corresponding labels
  
  % Example: [X, y] has 1000 training examples with labels and percent = 0.85
  %           -> X_train will have 850 examples
  %           -> X_test will have the other 150 examples

  % TODO: split_dataset implementation
  idx = randperm(length(y));
  y = y(idx);
  X = X(idx, :);
  nr = percent * length(y);
  
  X_train = X(1 : nr, 1 : columns(X));
  y_train = y(1 : nr);
  
  X_test = X(nr + 1 : rows(X), 1 : columns(X));
  y_test = y(nr + 1 : length(y));
endfunction
