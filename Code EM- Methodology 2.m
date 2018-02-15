
close all;
clear;
clc;

% initialize the parameters
alpha = [0.2,0.25,0.3,0.1,0.15];    % initial probability weight for each distribution
n = 1200;    % number of independent samples
iter = 200;    % number of iterations
K = 5;    % number of different distributions
N = 100;   % number of spacial intervals
loglike = zeros(1,iter+1);    % log_likelihood function

% set up the initial guess vectors & matrices
X = zeros(2,n);           % an array to store bivaraite normal samples up to n
est_mu = zeros(2,K);         % an array to store all estimated means from k distributions.
est_cov = zeros(2,2,K);        % an array to store all estimated std.dev from k distributions
est_alpha = zeros(1,K);          % a row vector to store each estimated weight of k distributions. 


color = hsv(K);            % an array stores the colormap. 
col_rg = zeros(n,3);     % an array to store the random generated row vectors from col, simulating n times.

% specifying true mean
mu = [7 10 12 8 13; 10 12 8 14 12];

% specifying true covariance matrix
cov = zeros(2,2,5);
cov(:,:,1) = [1 0.2;0.2 1.5];
cov(:,:,2) = [1.5 2;2 4];
cov(:,:,3) = [3 -0.5;-0.5 2];
cov(:,:,4) = [1.5 -1;-1 2];
cov(:,:,5) = [1 -1.5;-1.5 3];

% generate n multivariate normal distribution samples
pc = [ones(1,n*0.2) 2*ones(1,n*0.25) 3*ones(1,n*0.3) 4*ones(1,n*0.1) 5*ones(1,n*0.15)];
for jj = 1:n
    
    % simulate random integers from 1 to k, according to the corresponding
    % weight
    i = randi([1 1200],1,1); 
    ii = pc(i);
    
    % assigning each row of col_rg from random choice of row of col.
    col_rg(jj,:) = color(ii,:); 
    
    % fill in all columns of x by generated multivariate normal samples.
    X(:,jj) = mvnrnd(mu(:,ii), cov(:,:,ii)); 
end

 
% set up the initial estimates for mean, covariance matrix and weights
for jj = 1:K
    est_mu(:,jj) = [8+3*rand(1), 9+3*rand(1)];
    est_cov(:,:,jj) = [1 0;0 1];
    est_alpha(jj) = 1/K;
end

% iteration starts...
for t = 1:iter

figure(1);
% scatter plot the generated multivariate normal samples
hold off
scatter(X(1,:),X(2,:),15,col_rg,'filled');
hold on

% plot the true mean and the estimated mean;
plot(mu(1,:),mu(2,:),'r.','markersize',40);
plot(est_mu(1,:),est_mu(2,:),'b.','markersize',40);

%% E-Step

% prepare the range of the level curve of the density
[A,B] = meshgrid(linspace(4,16,N),linspace(4,16,N));
P_ = [A(:),B(:)]; 

% generate the density of the multivariate normal distribution
density_new = zeros(5,10000);
for j = 1:K
    density_new(j,:) = est_alpha(j)*mvnpdf(P_,est_mu(:,j)',est_cov(:,:,j))';
end

% determine a row vector of max value for each column and also 
% determine its corresponding row index for each column
[M,I] = max(density_new,[],1);

% sum each column of the array densityfun and put into row vector.
Z = sum(density_new,1);

% then shape the generated vector Z into matrix of dimension same as XX. 
Z = reshape(Z,size(A));
% also reshape index into matrix of dimesion same as XX.
I = reshape(I,size(A));

% contour plot the density function
contour(A,B,Z);
contour(A,B,I);

% title the plot with marking the time
title(sprintf('iteration # %d / %d Loglikelihood', t, iter));
pause(0.1)
drawnow;


% evaluate mixture model density: alpha(j)*density 
term = zeros(n,K);
for ii = 1:n
    for jj = 1:K
        term(ii,jj) = est_alpha(jj)*mvnpdf(X(:,ii),est_mu(:,jj),est_cov(:,:,jj));
    end
end

% evaluate conditional density P(i,j) at old time t
P = zeros(n,K);
for ii = 1:n
    for jj = 1:K
        P(ii,jj) = term(ii,jj)/sum(term(ii,:));
    end
end

% the expectation of the complete log-likelihood Q at time t
q = zeros(n,K);

for ii = 1:n
    for jj = 1:K
        q(ii,jj) = P(ii,jj)*log(term(ii,jj));
    end
end
Q = sum(q(:));

% evaluate the loglikelihood function
        loglike(t+1) = Q;

%% M-Step

% the updated estimates of weight at time t+1
est_alpha = sum(P,1)/n;

% the upadated estimates of mean at time t+1
for jj = 1:K
    est_mu(:,jj) = X*P(:,jj)/sum(P(:,jj));
end

% the updated estimates of covariance matrix at time t+1
for j = 1:K
    
    denom = sum(P(:,j));
    X_ = zeros(size(X));
    X__ = zeros(size(X));
    
    X__(1,:) = (X(1,:) - est_mu(1,j));
    X__(2,:) = (X(2,:) - est_mu(2,j));
    
    X_(1,:) = (X(1,:) - est_mu(1,j)).*P(:,j)';
    X_(2,:) = (X(2,:) - est_mu(2,j)).*P(:,j)';
    
    
    num = X_ * X__';
    est_cov(:,:,j) = num/denom;
end
 
% % plot after few iterations
% if t == 1 || t == 3 || t == 7 || t == 15 || t == 31 || t == 63
%     pause;
% end
        
% termination condition
if abs((loglike(t+1) - loglike(t))/loglike(t)) < 1e-8
    fprintf('the iteration process ends at # %d',t);
    break;
end

end

figure(2);

plot(loglike,'*');
title('loglikelihood function');


    
    