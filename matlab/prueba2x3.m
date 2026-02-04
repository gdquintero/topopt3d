nodes = 30;
I = eye(nodes * 2);
nele = 20;
ny = 5;
k = ny - 1;
K = zeros(nodes*2,nodes*2);
Ke = [1 1 1 1 1 1 1 1;2 2 2 2 2 2 2 2;3 3 3 3 3 3 3 3;4 4 4 4 4 4 4 4;5 5 5 5 5 5 5 5;6 6 6 6 6 6 6 6;7 7 7 7 7 7 7 7;8 8 8 8 8 8 8 8];
for i = 1:nele
    node = (floor((i-1)/k))*ny+mod(i-1,k)+1; 
    P = [I(:, 2*node - 1), I(:, 2*node), I(:, 2*(node + ny) - 1 ),  I(:, 2*(node + ny)), I(:, 2*(node + ny + 1) - 1), I(:, 2*(node + ny + 1)), I(:, 2*(node + 1) - 1), I(:, 2*(node + 1))];
    K = K +  P*Ke*P';
end

% disp(K);
% P1 = [I(:,1),I(:,2),I(:,7),I(:,8),I(:,9),I(:,10),I(:,3),I(:,4)];
% P2 = [I(:,3),I(:,4),I(:,9),I(:,10),I(:,11),I(:,12),I(:,5),I(:,6)];

spy(K);

% xlswrite("K2x3.xlsx", K);
% K = P1*Ke*P1';
% K = K + P2*Ke*P2'
