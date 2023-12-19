clc;clf;clear

%1.European Put Call, 2.Down In, 3. Down Out, 4. Up In, 5. Up Out, 6. Double In, 7. Double Out
OptionType = 5

S0 = 100; % price of the stock at time 0
V = 0.25; % volatility of the stock
K = 105; % strike price
H = 115; % barrier for single barrier options
U = 120; %Upper Barrier
L = 80; %Lower Barrier
r = 0.025; % risk free rate
T = 1; % maturity
NoPath = 500; % number of paths-simulated
NoSteps = 500; % number of monitoring steps
TI = 1/NoSteps; % time interval
DiscFac = exp(- r * T); %discount factor
S1 = zeros(1,NoSteps);%Path 1
S2 = zeros(1,NoSteps);%Arithmetic Partner Path
N = zeros(1,NoSteps);%NormalDist RN



% Define the parameters
n = NoSteps + 1; % Number of samples
dim = NoPath;   % Number of dimensions

% Generate Sobol sequence
sobolSeq = sobolset(dim);
sobolPoints = net(sobolSeq, n);

% Transform Sobol points to approximate normal distribution
normApprox = norminv(sobolPoints);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if OptionType == 1 %Euro Put Call
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0;

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
end
%Compute Payoff
CPayOff(i) = max(S1(i,NoSteps)-K,0);
PPayOff(i) = max(K-S1(i,NoSteps),0);

end

%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif OptionType == 2 %Down In
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0;

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
S2(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) - V * sqrt(TI) * normApprox(i,j));
end

if min(S1(i,:)) < H 
CPayOff1(i) = max(S1(i,NoSteps)-K,0);
PPayOff1(i) = max(K-S1(i,NoSteps),0);
else
CPayOff1(i) = 0;
PPayOff1(i) = 0;
end

if min(S2(i,:)) < H 
CPayOff2(i) = max(S2(i,NoSteps)-K,0);
PPayOff2(i) = max(K-S2(i,NoSteps),0);
else
CPayOff2(i) = 0;
PPayOff2(i) = 0;
end

CPayOff(i) = (CPayOff1(i) + CPayOff2(i))/2;
PPayOff(i) = (PPayOff1(i) + PPayOff2(i))/2;
end
%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif OptionType == 3 %Down In
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0;

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
S2(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) - V * sqrt(TI) * normApprox(i,j));
end

if min(S1(i,:)) > H 
CPayOff1(i) = max(S1(i,NoSteps)-K,0);
PPayOff1(i) = max(K-S1(i,NoSteps),0);
else
CPayOff1(i) = 0;
PPayOff1(i) = 0;
end

if min(S2(i,:)) > H 
CPayOff2(i) = max(S2(i,NoSteps)-K,0);
PPayOff2(i) = max(K-S2(i,NoSteps),0);
else
CPayOff2(i) = 0;
PPayOff2(i) = 0;
end

CPayOff(i) = (CPayOff1(i) + CPayOff2(i))/2;
PPayOff(i) = (PPayOff1(i) + PPayOff2(i))/2;
end
%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif OptionType == 4 %Up In
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0; 

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
S2(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) - V * sqrt(TI) * normApprox(i,j));
end

if max(S1(i,:)) > H 
CPayOff1(i) = max(S1(i,NoSteps)-K,0);
PPayOff1(i) = max(K-S1(i,NoSteps),0);
else
CPayOff1(i) = 0;
PPayOff1(i) = 0;
end

if max(S2(i,:)) > H 
CPayOff2(i) = max(S2(i,NoSteps)-K,0);
PPayOff2(i) = max(K-S2(i,NoSteps),0);
else
CPayOff2(i) = 0;
PPayOff2(i) = 0;
end

CPayOff(i) = (CPayOff1(i) + CPayOff2(i))/2;
PPayOff(i) = (PPayOff1(i) + PPayOff2(i))/2;
end
%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif OptionType == 5 %Up Out
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0;

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
S2(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) - V * sqrt(TI) * normApprox(i,j));
end

if max(S1(i,:)) < H 
CPayOff1(i) = max(S1(i,NoSteps)-K,0);
PPayOff1(i) = max(K-S1(i,NoSteps),0);
else
CPayOff1(i) = 0;
PPayOff1(i) = 0;
end

if max(S2(i,:)) < H 
CPayOff2(i) = max(S2(i,NoSteps)-K,0);
PPayOff2(i) = max(K-S2(i,NoSteps),0);
else
CPayOff2(i) = 0;
PPayOff2(i) = 0;
end

CPayOff(i) = (CPayOff1(i) + CPayOff2(i))/2;
PPayOff(i) = (PPayOff1(i) + PPayOff2(i))/2;
end
%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif OptionType == 6 % Double Out
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0; 

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
S2(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) - V * sqrt(TI) * normApprox(i,j));
end

if min(S1(i,:)) > L & max(S1(i,:)) < H 
CPayOff1(i) = max(S1(i,NoSteps)-K,0);
PPayOff1(i) = max(K-S1(i,NoSteps),0);
else
CPayOff1(i) = 0;
PPayOff1(i) = 0;
end

if min(S2(i,:)) > L & max(S2(i,:)) < H 
CPayOff2(i) = max(S2(i,NoSteps)-K,0);
PPayOff2(i) = max(K-S2(i,NoSteps),0);
else
CPayOff2(i) = 0;
PPayOff2(i) = 0;
end

CPayOff(i) = (CPayOff1(i) + CPayOff2(i))/2;
PPayOff(i) = (PPayOff1(i) + PPayOff2(i))/2;
end
%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
elseif OptionType == 7 % Double In
for i = 1:NoPath
%Set Initial Values
S1(i,1) = S0;
S2(i,1) = S0;

for j = 2:NoSteps
%Compute Path
S1(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) + V * sqrt(TI) * normApprox(i,j));
S2(i,j) = S1(i,j-1)*exp((r - 0.5*V^2)*(TI) - V * sqrt(TI) * normApprox(i,j));
end

if min(S1(i,:)) < L | max(S1(i,:)) > H 
CPayOff1(i) = max(S1(i,NoSteps)-K,0);
PPayOff1(i) = max(K-S1(i,NoSteps),0);
else
CPayOff1(i) = 0;
PPayOff1(i) = 0;
end

if min(S2(i,:)) < L | max(S2(i,:)) > H  
CPayOff2(i) = max(S2(i,NoSteps)-K,0);
PPayOff2(i) = max(K-S2(i,NoSteps),0);
else
CPayOff2(i) = 0;
PPayOff2(i) = 0;
end

CPayOff(i) = (CPayOff1(i) + CPayOff2(i))/2;
PPayOff(i) = (PPayOff1(i) + PPayOff2(i))/2;
end
%Option Price
C = mean(CPayOff)*exp(-r*T)
P = mean(PPayOff)*exp(-r*T)

end

