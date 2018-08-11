function [A_tilde_disk, X, xi_disk] = monte_carlo(N)

X = rand(N,2);
xi_disk = sqrt(sum((X-repmat([0.5 0.5],N,1)).*(X-repmat([0.5 0.5],N,1)),2))<0.5;
    
inside_count = sum(xi_disk);
    
A_tilde_disk = inside_count / N;

end