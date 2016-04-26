N = 6;

w_state(:,1) = (rand(N,1) - 0.5) .* pi; %Initialize the phase for each oscillator
w_intri = (rand(N,1) - 0.5) .* pi/2;

coupling = rand(N,N);
coupling = coupling .* coupling > 0.8;

t = linspace(0,10,0.01);

for ii = 2:1000
    for jj = 1:N
        for kk = 1:N
            w_state(jj,ii) = w_intri(jj) + 1/(0.1*N) * sum(sin(w_state(jj,ii-1) - w_state(kk,ii-1)));
        end
    end
end

%%
%Plot it
figure;
for jj = 1:N
    plot(w_state(jj,:));hold on;
end