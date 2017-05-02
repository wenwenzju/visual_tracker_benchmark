function h2 = histogram2(x,y,n)
minx = min(x);
miny = min(y);
maxx = max(x)+1;
maxy = max(y)+1;
xx = linspace(minx, maxx, n+1);
yy = linspace(miny, maxy, n+1);

h = zeros(n,n);
for i = 1:n
    for j = 1:n
        tmp = (x>=xx(i))&(x<xx(i+1))&(y>=yy(j))&(y<yy(j+1));
        h(i,j) = sum(tmp);
    end
end

h2.Values = h';
h2.x = (xx(1:n)+xx(2:end))/2;