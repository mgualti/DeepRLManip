% 
% Illustrates the process of calculating grasp candidates and getting
% ground truth labels using a mesh.
% 
function exampleGraspCandidates

    close all;
    
    DEBUG = {};
    DEBUG = [DEBUG, {'clsPtsHands.getGraspCandidates'}];

    % Both hand parameters and image parameters must be the same here as
    % they were during training
    handparams = handParameters();
    handparams.fw = 0.01;
    handparams.handOD = 0.12; % 0.09?
    handparams.handDepth = 0.06;
    handparams.handHeight = 0.02;
    
    categoryroot = './data/MATfiles/';
    folderset = {'advil'};

    % Get random point cloud from an object in <folderset>. You should be
    % able to use any point cloud here, but make sure you get the
    % camsource(s) correct.
    useparfor = 0;
    tc = clsTrainCaffe(categoryroot,folderset,useparfor,0);  
    singleDual = 0; % get a random single (not stereo) cloud
    [p, mesh] = tc.getRandomCloud(singleDual);
    
    % Get some grasp candidates
    objUID = 'foo';
    camSet = [0];
    numSamples = 10;
    hands = clsPtsHands(p,handparams,objUID,camSet,DEBUG);
    hands = hands.subSample(numSamples);
    hands = hands.getGraspCandidates(p);
    
    % If we have the mesh corresponding/aligned with to the cloud, we can
    % also calculate ground truth
    hands = hands.calculateLabels(mesh);
    [handsListPos, ~, handsListHalf] = hands.getAntipodalHandList();    
    
    % Plot detected positives
    figure;
    p.plot(); hold on;
    hands.plotHandList(handsListPos);
    lightangle(-45,30);
    title('positive grasp instances');
    
    
end

