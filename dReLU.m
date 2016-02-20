function output = dReLU(x, w, b)

% The derivative of the ReLU (leaky) function.

output = max(x*w+b > 0, 0.01);

end