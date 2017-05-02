function residual = gaussian_fit2(x,y,p)
miux = mean(x);
miuy = mean(y);
sigmax = std(x);
sigmay = std(y);
cor = corrcoef(x,y);
cor = cor(1,2);
f = gaussian2(x,y,miux,miuy,sigmax,sigmay,cor);
p = p/sum(p);
f = f/sum(f);
residual = (p-f).^2;
residual = sqrt(residual);
residual = sum(residual);

function f = gaussian2(x,y, miux, miuy, sigmax, sigmay, cor)
f = 1.0/(2*pi*sigmax*sigmay*sqrt(1-cor*cor))*exp(-0.5/(1-cor*cor)*((x-miux).^2/sigmax^2)+(y-miuy).^2/sigmay^2-2*cor*(x-miux).*(y-miuy)/(sigmax*sigmay));