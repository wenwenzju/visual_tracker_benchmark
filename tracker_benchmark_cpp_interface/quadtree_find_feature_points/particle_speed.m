ps = importdata('particle_states.txt');
np = 10;
[nimg,nstate] = size(ps);
nimg = nimg/np;
x = ps(:,1);
x = reshape(x,np,[]);
y = ps(:,2);
y = reshape(y,np,[]);
dx = x(:,2:end)-x(:,1:end-1);
dy = y(:,2:end)-y(:,1:end-1);
dv = sqrt(dx.^2+dy.^2);
figure
plot(2:nimg, dx, '-^');
figure
plot(2:nimg, dy, '-.');
figure
plot(2:nimg, dv, '-.');
%%
fid = fopen('feature_points.txt');
dl = fgetl(fid);
nsigma = 1;
while ischar(dl)
    points1 = sscanf(dl,'%f');
    dl = fgetl(fid);
    points2 = sscanf(dl, '%f');
    dp = points2-points1;
    dpx = dp(1:2:end);
    dpy = dp(2:2:end);
    dpv = sqrt(dpx.^2+dpy.^2);
    [dpv,idx] = sort(dpv);
    cor = 1:length(dpv);
    plot(cor, dpv,'-^');
    m = mean(dpv);
    s = std(dpv);
    outlier = find(dpv < m-nsigma*s | dpv > m+nsigma*s);
    hold on;
    plot(cor(outlier), dpv(outlier),'r*');
    hold off;
%     [~,idx] = sort(dpx);
%     plot(1:length(dpx), dpx(idx),'-^');
    pause
    dl = fgetl(fid);
end