clear ; close all; clc

%Parameters

input_layer_size  = 400;  % 20x20 Images
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('data1.mat');
m = size(X, 1);

% Randomly display 100 data points 
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));



fprintf('\n Initializing Parameters...\n')

%random initialize parameters using randInit.m
iTheta1 = randInit(input_layer_size,hidden_layer_size);
iTheta2 = randInit(hidden_layer_size,num_labels);

%unrolling parameters
iParams = [iTheta1(:);iTheta2(:)];

%--Training
fprintf('\nTraining Neural Network\n');
options = optimset('MaxIter',50); % max iteration
lambda = 1; % regularization amount

% short hand for the cost function
Costf = @(p) CostFunction(p,input_layer_size,hidden_layer_size,num_labels,X,y,lambda);

[params,cost] = fmincg(Costf,iParams,options);
%get Theta back
Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
  
  %Visualize hidden units
  fprintf('\nVisualizing hidden layers \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

s = input('Paused - press enter to continue','s');



%visualise examples
rp = randperm(m);
for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end
