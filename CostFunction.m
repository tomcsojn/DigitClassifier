function [J grad] = CostFunction(params,inputlayer_size,hiddenlayer_size,num_labels,X,y,lambda)
  
  Theta1 = reshape(params(1:hiddenlayer_size * (inputlayer_size + 1)), ...
                 hiddenlayer_size, (inputlayer_size + 1));

  Theta2 = reshape(params((1 + (hiddenlayer_size * (inputlayer_size + 1))):end), ...
                 num_labels, (hiddenlayer_size + 1)); 
  
  
  m = size(X,1);
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  
  %feedforward
  
  %add bias unit to dataset
  X = [ones(m, 1) X];
  
  a2 = sigmoid(Theta1*X'); %
  a2 = [ones(m,1) a2']; %add bias unit
  h = sigmoid(Theta2*a2')'; 
  %convert y(numbers) to yv(vector)
  
  yv = bsxfun(@eq,y(:),(1:num_labels))';
##  yv = zeros(m,num_labels);
##  for i = 1:m
##  yv(i,y(i)) = 1;
##  endfor
##  yv = yv';
  %cost function
  J = (1/m)*trace(-yv*log(h)-(1-yv)*log(1-h));
  %add regularization
##t1 = 0;
##t2 = 0;
##for i=1:size(Theta1,1)
## for j=2:size(Theta1,2)
##   t1 = t1+Theta1(i,j)^2;
## endfor
##endfor
##for i=1:size(Theta2,1)
## for j=2:size(Theta2,2)
##   t2 = t2+Theta2(i,j)^2;
## endfor
##endfor
t1 = trace(Theta1(:,2:end).^2);
t2 = trace(Theta2(:,2:end).^2);
J = J+ (lambda/(2*m))*(t1+t2);
  %Partial Derivatives
  d3 = h-yv';
  d2 = Theta2'*d3'.*[ones(1,m);sigmoidGradient(Theta1*X')];
  d2 = d2(2:end,:); % need to remove the bias errors
  Theta1g = (1/m)*(d2*X);
  Theta2g = (1/m)*(d3'*a2);
  %regularization
  Theta1g(:,2:end) = Theta1g(:,2:end)+(lambda/m)*Theta1(:,2:end);
  Theta2g(:,2:end) += (lambda/m)*Theta2(:,2:end);
  
  %unroll gradients
  grad = [Theta1g(:);Theta2g(:)];
   
endfunction
