clear;
clear all
K=5;
mu(:,:,1) = [7 10];       % Mean
S(:,:,1)=[1,.2;0.2,1.5];
mu(:,:,2) = [10 12];
S(:,:,2) = [1, 1.5; 1.5, 3];
mu(:,:,3) = [12 18];
S(:,:,3) = [3, -.5; -.5, 2];
mu(:,:,4) = [8 14];
S(:,:,4) = [1.5, -1; -1, 2];
mu(:,:,5) = [13 12];
S(:,:,5) = [1, -1.5; -1.5, 3];
n=1200;
p=[0.2, 0.25, 0.3, 0.1, 0.15];
pc=cumsum(p(1:K-1)); %Calculate X values under mvn process
q=ones(n,K);
col=hsv(K);         
colcat=ones(n,3);
for i=1:n
    j=sum(pc<rand())+1;
    colcat(i,:)=col(j,:);
    x(:,i)=mvnrnd(mu(:,:,j),S(:,:,j));
end
hX=x';
loglike=10000000000000000000;
prevloglike=1000000000000000000000000;
linespec = {'o', 'o', 'o','o', 'o', 'o'}; % define your linespecs in a cell array

%Initialization of the parameters.

m = size(hX, 1);
index = randperm(m);
hmu = hX(index(1:K), :)+rand();
hS=[];      %all h std dev

for j=1:K
    hS{j}=[1 0;0 1];
end

alpha = ones(1, K) * (1 / K); %Equal probablity to each sample.
hmu1=hmu';
wp = zeros(m, K); % Row denotes sample values, column denotes the sample.

for (iteration = 1:10)   %Expectation - STEP
    pdf = zeros(m, K); 
    for (j = 1 : K)
        pdf(:, j) = Gaussian_value(hX, hmu(j, :), hS{j}); 
    end
    pdf_w = bsxfun(@times, pdf, alpha);
    wp = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2)); 
    for i=1:n
        for j=1:K
            q(i,j)=alpha(j)*mvnpdf(x(:,i),hmu1(:,j),hS{j});
        end
    end

    prevlogike=loglike;
    loglike=sum(log(sum(q,2)));   

%Maximization - STEP
    prevhmu = hmu;    
    for (j = 1 : K)
        alpha(j) = mean(wp(:, j), 1);
        hmu(j, :) = wAvg(wp(:, j), hX);
        hS_K = zeros(n, n);%Caluclate sigma of each sample by taking W.A. of the sigma
        Xmu = bsxfun(@minus, hX, hmu(j, :));
        for (i = 1 : m)
            hS_K = hS_K + (wp(i, j) .* (Xmu(i, :)' * Xmu(i, :)));
        end
        hS{j} = hS_K ./ sum(wp(:, j));
    end

    if abs((loglike-prevloglike)/prevloglike)<0.000000001
        break;
    end        
end

figure(1);
scatter(x(1,:),x(2,:),'o');

hold on
set(gcf,'color','white') 

plot(hmu(:,1), hmu(:,2),'k*','markersize',20)

for i=1:K
    plot(mu(:,1,i),mu(:,2,i), 'ko','markersize',20);
end

v = linspace(4, 24, 100);
[XX1 YY1] = meshgrid(v, v);
gridX1 = [XX1(:), YY1(:)];

for i=1:K
    z(:,:,i) = Gaussian_value(gridX1, hmu(i,:), hS{i});
end

for i=1:K
    Z(:,:,i) = reshape(z(:,:,i), 100, 100);
end

for i=1:K
    [C, h] = contour(v, v, Z(:,:,i));
end

axis([4 24 4 24])

title(['EM ALGORITHM PLOT - ', 'Iteration ' ,num2str(iter), ' Log-Likelihood ' ,num2str(loglike)]);

function [ pdf_value ] = Gaussian_value(hX, mu, hS)
    n = size(hX, 2);
    Diff = bsxfun(@minus, hX, mu);
    pdf_value = 1 / sqrt((2*pi)^n * det(hS)) * exp(-1/2 * sum((Diff * inv(hS) .* Diff), 2));
end

function [ value ] = wAvg(weights, values)
    value = weights' * values;
    value = value ./ sum(weights, 1);
end
