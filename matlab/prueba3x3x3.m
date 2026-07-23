
I = eye(192);
nele = 27;
nelx = 3;
nely = 3;
ny = 4;
nx = 4;
nodesNumber = nx * ny;
% k = ny - 1;
% K1 = zeros(50,50);
% K2 = zeros(50,50);
K = zeros(192, 192);
Ke = [1 1 1 1 1 1;2 2 2 2 2 2 ;3 3 3 3 3 3;4 4 4 4 4 4;5 5 5 5 5 5;6 6 6 6 6 6];
KeRec = ones(24);
floor = 0;
% node = 0;
for i = 1:nele
    if mod(i, nelx) ~= 0
        node = i + fix(i/nelx) + floor*nx;
    else
        node = i + fix(i/nelx) + floor*nx  - 1;
    end

    if (nelx * nely * (floor + 1)) == i
        floor = floor + 1;
    end

    % node = (floor((i-1)/k))*ny+mod(i-1,k)+1;
    % P1 = [I(:, 2*node - 1), I(:, 2*node), I(:, 2*(node + ny) - 1 ),  I(:, 2*(node + ny)), I(:, 2*(node + ny + 1) - 1), I(:, 2*(node + ny + 1)) ];
    % P2 = [I(:, 2*node - 1), I(:, 2*node), I(:, 2*(node + ny + 1) - 1), I(:, 2*(node + ny + 1)),I(:, 2*(node + 1) - 1), I(:, 2*(node + 1))];

    P = [I(:, 2*node - 1), I(:, 2*node), I(:, 2*(node + 1) - 1), I(:, 2*(node + 1)), I(:, 2*(node + ny) - 1 ),  I(:, 2*(node + ny)), I(:, 2*(node + ny + 1) - 1), I(:, 2*(node + ny + 1))];
    disp(node)
    disp(3*(node  + nodesNumber + ny + 1))

    P3d = [I(:, 3*node - 2), I(:, 3*node - 1), I(:, 3*node), ...
            I(:, 3*(node + ny) - 2 ), I(:, 3*(node + ny) - 1 ), I(:, 3*(node + ny)), ...
             I(:, 3*(node + ny + 1) - 2 ), I(:, 3*(node + ny + 1) - 1 ), I(:, 3*(node + ny + 1)), ...
             I(:, 3*(node + 1) - 2), I(:, 3*(node + 1) - 1), I(:, 3*(node + 1)), ...
             I(:, 3*(node + nodesNumber) - 2), I(:, 3*(node + nodesNumber) - 1), I(:, 3*(node + nodesNumber)), ...
            I(:, 3*(node + nodesNumber + ny) - 2 ), I(:, 3*(node + nodesNumber + ny) - 1 ), I(:, 3*(node + nodesNumber + ny)), ...
             I(:, 3*(node + nodesNumber + ny + 1) - 2 ), I(:, 3*(node + nodesNumber + ny + 1) - 1 ), I(:, 3*(node + nodesNumber + ny + 1)), ...
             I(:, 3*(node + nodesNumber + 1) - 2), I(:, 3*(node + nodesNumber + 1) - 1), I(:, 3*(node + nodesNumber + 1))];
    % K1 = K1  + P1*Ke*P1';
    % K2 = K2  + P2*Ke*P2';
    K = K + P3d*KeRec*P3d';
    % K = K + P*KeRec*P';
end

% disp(K);
% P1 = [I(:,1),I(:,2),I(:,7),I(:,8),I(:,9),I(:,10),I(:,3),I(:,4)];
% P2 = [I(:,3),I(:,4),I(:,9),I(:,10),I(:,11),I(:,12),I(:,5),I(:,6)];

% spy(K2);

spy(K)

xlswrite("Ktrian3x3x3.xlsx", K);
% xlswrite("KOtrian5x5.xlsx", K1)
% xlswrite("KEtrian5x5.xlsx", K2)
% xlswrite("Ktrian5x5.xlsx", K1 + K2);

% spy(K1 + K2)
