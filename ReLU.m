function output = ReLU(x, w, b)

y = x*w+b;
output = max(y,y*0.01);

end