function output = Sigmoid(x, w, b)

% The Sigmoid activation function

y = x*w+b;
output = 1/(1+exp(-y));

end