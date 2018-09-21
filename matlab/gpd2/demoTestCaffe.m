% 
% This file demos the typical process for using caffe to detect grasps on a
% novel point cloud.
%
% Run this file after you have already run trainCaffe.m
% 
% Since this file loads a random point cloud from bigbird, you need to have
% categoryroot and folderset the same as they were in createImages.m
% 
% handparams and imageparams must be the same as they were during training
% 
function demoTestCaffe

    close all;
    
    % Both hand parameters and image parameters must be the same here as
    % they were during training
    handparams = handParameters();
    handparams.fw = 0.01;
    handparams.handOD = 0.12; % 0.09?
    handparams.handDepth = 0.06;
    handparams.handHeight = 0.02;
    
    imageparams = imageParameters();
    imageparams.imageOD = 0.10;
    imageparams.imageDepth = 0.06;
    imageparams.imageHeight = 0.02;
    imageparams.imageSize = 60;
    
    categoryroot = './data/MATfiles/';
    folderset = {'advil'};

    % Get random point cloud from an object in <folderset>. You should be
    % able to use any point cloud here, but make sure you get the
    % camsource(s) correct.
    useParfor = 1;
    tc = clsTrainCaffe(categoryroot,folderset,useParfor,'');    
    p = tc.getRandomCloud();
    
    % Get grasp candidates
    objUID = 'foo';
    camSet = [0];
    numSamples = 10;
    hands = clsPtsHands(p,handparams,objUID,camSet,0);
    hands = hands.subSample(numSamples); % number of samples to use in hand detection
    hands = hands.getGraspCandidates(p);
    
    % Plot grasp candidates
    figure;
    p.plot();
    hold on;
    hands.plotHandList(hands.getAllHands());
    lightangle(-45,30);
    title('grasp candidates');
    
    % Calculate grasp images
    hands = hands.setImageParams(imageparams);
    hands = hands.calculateOcclusion(p);
    hands = hands.calculateImages(p,hands.getAllHands(),0);
    hands = hands.clearCloud();
    hands = hands.clearHandSetListHoods();
    
    % Write grasp images to caffe folder
    l = clsLearning();
    l = l.importFromHands(hands);
    idx = 1:l.num();
    
    predictionList = l.predictGrasps(idx, './deploy_oldversion.prototxt', './data/CAFFEfiles/lenet_iter_1000.caffemodel', 0);
    
%     l.writeToCAFFE('./data/temp',idx,idx);
%     pause(0.5);
%     system('python deploy_classifier.py');
%     pause(0.5);
%     predictionList = l.readFromCAFFE('./data/temp');
    
    graspRank = predictionList(:,2) - predictionList(:,1);
    
    rankedGrasps = sortrows([1:size(graspRank,1); graspRank']',2);
    idxTopFive = rankedGrasps(end-4:end,1);
    labels = zeros(1,size(hands.getAllHands(),1));
    labels(idxTopFive) = 1;
    hands = hands.importAntipodalLabels(hands.getAllHands(),labels);
    
    % Plot predicted positives
    figure;
    p.plot();
    hold on;
    hands.plotHandList(hands.getAntipodalHandList());
    lightangle(-45,30);
    title('top 5 predicted grasps');
    
end
