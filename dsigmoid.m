function output = dsigmoid(x, w, b)

% The derivative of the Sigmoid function.

output = sigmoid(x,w,b)*(1-sigmoid(x,w,b));

end