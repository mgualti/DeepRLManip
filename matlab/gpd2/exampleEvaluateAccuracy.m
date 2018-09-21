% 
% Evaluate grasp prediction accuracy in a deployment-like scenario.
% 
function exampleEvaluateAccuracy

    close all;
    
    % Both hand parameters and image parameters must be the same here as
    % they were during training
    [handparams, imageparams] = getParams();
    
    categoryroot = './data/MATfiles/';
    folderset = {'advil'};

    % Get random point cloud and mesh
    useParfor = 1;
    tc = clsTrainCaffe(categoryroot,folderset,useParfor,'');    
%     [p, mesh] = tc.getRandomCloud(1); % stereo cloud
    [p, mesh] = tc.getRandomCloud(0); % single cloud
    
    % Get grasp candidates
    objUID = 'foo';
    camSet = [0];
    numSamples = 1000;
    hands = clsPtsHands(p,handparams,objUID,camSet,0);
    hands = hands.subSample(numSamples); % number of samples to use in hand detection
    hands = hands.getGraspCandidates(p);

    % Get ground truth labels using full mesh
    hands = hands.calculateLabels(mesh);
    [handsListPos, ~, handsListHalf] = hands.getAntipodalHandList();
    handsListNeg = setdiff(hands.getAllHands(),handsListPos,'rows');
    hands = hands.balance(handsListPos, handsListNeg);
    
    % Calculate grasp images
    hands = hands.setImageParams(imageparams);
    hands = hands.calculateOcclusion(p);
    hands = hands.calculateImages(p,0);
    hands = hands.clearCloud();
    hands = hands.clearHandSetListHoods();
    
    % Get predictions from caffe
    l = clsLearning();
    l = l.importFromHands(hands);
    pause(0.5); % without this pause, I seem to get error when running the prediction in the next line
    labelsPredicted = l.getPredictions('./deploy_oldversion.prototxt', './data/CAFFEfiles/lenet_iter_1000.caffemodel', 0);
    pause(0.5); 
     
    % Ground truth
    labelsGroundTruth = l.labels;
    
%     [labelsPredicted;labelsGroundTruth]
    accuracy = sum(labelsPredicted == labelsGroundTruth) / size(labelsPredicted,2)
    
end
