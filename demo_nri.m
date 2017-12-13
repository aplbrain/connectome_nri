% demo_nri.m

% Define count table.
% First row and column are deletions and insertions, respectively
C = [0,   100,    15,    10,   200;
     10,    1,    10,   300,    20;
     5,    10,   100,     5,    10];


fprintf('\n=========================\n');
fprintf('Results from nri()\n');
fprintf('=========================\n');
[nriNet, precNet, recallNet, nriNeur, precNeur, recallNeur] = nri(C);
fprintf('nriNet = %f\n',nriNet);
fprintf('precNet = %f\n',precNet);
fprintf('recallNet = %f\n',recallNet);


fprintf('\n=========================\n');
fprintf('Results from nri_slow()\n');
fprintf('=========================\n');
[nriNet, precNet, recallNet, nriNeur, precNeur, recallNeur] = nri_slow(C);
fprintf('nriNet = %f\n',nriNet);
fprintf('precNet = %f\n',precNet);
fprintf('recallNet = %f\n',recallNet);

fprintf('\n');
