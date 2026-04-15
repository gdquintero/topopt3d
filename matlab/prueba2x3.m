I = eye(50);
nele = 16;
ny = 5;
k = ny - 1;
K1 = zeros(50,50);
K2 = zeros(50,50);
Ke = [1 1 1 1 1 1;2 2 2 2 2 2 ;3 3 3 3 3 3;4 4 4 4 4 4;5 5 5 5 5 5;6 6 6 6 6 6];
for i = 1:nele
    node = (floor((i-1)/k))*ny+mod(i-1,k)+1;
    P1 = [I(:, 2*node - 1), I(:, 2*node), I(:, 2*(node + ny) - 1 ),  I(:, 2*(node + ny)), I(:, 2*(node + ny + 1) - 1), I(:, 2*(node + ny + 1)) ];
    P2 = [I(:, 2*node - 1), I(:, 2*node), I(:, 2*(node + ny + 1) - 1), I(:, 2*(node + ny + 1)),I(:, 2*(node + 1) - 1), I(:, 2*(node + 1))];
    K1 = K1  + P1*Ke*P1';
    K2 = K2  + P2*Ke*P2';
end

% disp(K);
% P1 = [I(:,1),I(:,2),I(:,7),I(:,8),I(:,9),I(:,10),I(:,3),I(:,4)];
% P2 = [I(:,3),I(:,4),I(:,9),I(:,10),I(:,11),I(:,12),I(:,5),I(:,6)];

% spy(K2);

% xlswrite("KOtrian5x5.xlsx", K1)
% xlswrite("KEtrian5x5.xlsx", K2)
% xlswrite("Ktrian5x5.xlsx", K1 + K2);

spy(K1 + K2)