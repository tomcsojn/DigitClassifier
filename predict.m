function p = predict(Theta1,Theta2,X)
  %Predict the label of the input X
  m = size(X,1);
  num_labels = size(Theta2,1);
  
  p = zeros(size(X,1),1);
  
  a1 = sigmoid([ones(m,1) X]*Theta1');
  a2 = sigmoid([ones(m,1) a1]*Theta2');
  [tmp,p] = max(a2,[],2);
  
endfunction
