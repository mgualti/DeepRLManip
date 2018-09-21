% Samples and classifies grasps in a point cloud.
%  Input cloud: 3xn point cloud for grasp detection.
%  Input viewPoints: 3xm points for each m viewpoints the cloud was taken
%    from.
%  Input viewPointIndices: Index into the point cloud where each of the
%    view points starts.
%  Input nSamples: Scalar integer number of points in cloud to sample.
%  Input scoreThresh: Scalar acceptance threshold for high scoring grasps.
%  Input plotMode: 1x5 bitmap for which plots to show.
%  Input gpuId: Integer indicating the system GPU to use for caffe.
%  Output: grasps: Data structure with grasp poses and grasp images.
function grasps = DetectGrasps(cloud, viewPoints, viewPointIndices, ...
    nSamples, scoreThresh, plotBitmap, gpuId)
    
    % 1. Setup.
    
    if plotBitmap(1)
        tic;    
        disp(['Detecting grasps with ' num2str(nSamples) ...
          ' samples in cloud with ' num2str(size(cloud,2)) ' points.']);
    end
    
    if numel(viewPointIndices) > 1 && size(viewPointIndices, 2) == 1
        viewPointIndices = viewPointIndices';
    end
  
    % 2. Parameters.
    
    deployFile = '/home/mgualti/Projects/gpd2_rave_blocks/data/CAFFEfiles/deploy.prototxt';
    caffeModelFile = '/home/mgualti/Projects/gpd2_rave_blocks/data/CAFFEfiles/lenet_iter_30000.caffemodel';
    
    %curveAxisProb = 0.33;
    curveAxisProb = 1.0;
    [handparams, imageparams] = getParams();
    
    % 3. Point cloud processing.
    
    p = clsPts();
    p = p.addCamPos(viewPoints);
    camsource = false(size(viewPoints, 2), size(cloud, 2));
    viewPointIndices = [viewPointIndices, size(cloud, 2)+1];
    for idx=1:size(viewPoints, 2)
        camsource(idx, viewPointIndices(idx):viewPointIndices(idx+1)-1) = true;
    end
    p = p.addPts(cloud, camsource);
    p = p.voxelize(0.002);
    
    % 4. Get grasp candidates.
    
    objUID = 'DetectGraspsObj';
    hands = clsPtsHands(p, handparams, objUID, (0), 0);
    hands = hands.subSample(nSamples);
    hands = hands.getGraspCandidates(p, curveAxisProb);
    
    % plot grasp candidates
    if plotBitmap(2)
        figure; hold('on');
        p.plot('k');
        hands.plotHandList(hands.getAllHands());
        lightangle(-45, 30);
        title('grasp candidates');
    end
    
    % 5. Calculate grasp images.
    
    hands = hands.setImageParams(imageparams);
    hands = hands.calculateImages(p); % longest
    
    % 6. Run Caffe and label grasps.
    
    l = clsLearning();
    l = l.importFromHands(hands);
    idx = 1:l.num();
    
    predictionList = l.getScores(deployFile, caffeModelFile, gpuId, idx);
    graspRank = predictionList(:,2) - predictionList(:,1);
    
    labels = graspRank >= scoreThresh;
    hands = hands.importAntipodalLabels(hands.getAllHands(), labels);
    antipodalHandList = hands.getAntipodalHandList();
    
    if plotBitmap(1)
        nOriginal = num2str(length(labels));
        nFiltered = num2str(length(labels)-sum(labels));
        nGrasps = num2str(sum(labels));
        disp(['Originally ' nOriginal ', filtered ' nFiltered ...
            ' below threshold.']);
        disp(['Now have ' nGrasps '.']);
    end
    
    % plot predicted positives
    if plotBitmap(3)
        figure; hold('on'); p.plot('k');
        hands.plotHandList(antipodalHandList);
        lightangle(-45, 30);
        title('Predicted grasps');
    end
    
    % 8. Package result.
    
    grasps = cell(1,size(antipodalHandList, 1)); scores = graspRank(labels);
    for idx=1:size(antipodalHandList, 1)
        handIdx = antipodalHandList(idx, 1);
        orientIdx = antipodalHandList(idx, 2);
        hs = hands.handSetList{handIdx};
        hand = hs.Hands{orientIdx};
        grasps{idx} = PackageHand(hand, scores(idx));
    end
    
    if plotBitmap(1)
        toc
    end
end

% Puts hand into the structure that the calling program will recognize.
function grasp = PackageHand(hand, score)
    grasp = struct();
    [grasp.center, grasp.bottom, grasp.top] = hand.getHandParameters();
    grasp.approach = hand.F(:,1);
    grasp.axis = hand.F(:,3);
    grasp.binormal = hand.F(:,2);
    grasp.height = hand.handparams.handHeight;
    grasp.width = hand.width;
    grasp.score = score;
end