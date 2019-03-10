function W = randInit(in,out)
  %randomly initialize the weights for a layer with "in" incoming connections and "out" outgoing connections.
 W = zeros(out,1+in);
 epsilon_init = 0.12; 
W = rand(out, 1 + in) * 2 * epsilon_init - epsilon_init; 
endfunction
