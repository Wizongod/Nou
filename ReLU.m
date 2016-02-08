function output = ReLU(x, w, b)

% The ReLU (leaky) activation function.

y = x*w+b;
output = max(y,y*0.01);

end