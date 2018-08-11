function [] = plot_monte_carlo(N, X, xi_disk)

inside_count = sum(xi_disk);
    

plot_geometry;

hold on;

C = zeros(N,3);
C(xi_disk,:) = repmat([0 0.75 0],inside_count,1);
C(~xi_disk,:) = repmat([1 0 0],N-inside_count,1);

scatter(X(:,1),X(:,2),35,C, 'filled');

hold off;

end