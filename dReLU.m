function output = dReLU(x, w, b)

output = max(x*w+b > 0,0.01);

end