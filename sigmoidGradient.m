function g = sigmoidGradient(z)
  %returning the gradient of the sigmoid function for both vectors and matrixes
  g = zeros(size(z));
  g = sigmoid(z).*(1-sigmoid(z));  
endfunction
