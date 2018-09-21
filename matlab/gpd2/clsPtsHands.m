classdef clsPtsHands < clsPtsFrenet
    
    properties
        DEBUG;
        handSetList; % HandSet list
        handparams;
        imageparams;
        objUID;
        camSet;
    end
    
    methods
        

        % Constructor
        % input: pin -> clsPts/clsPtsHood/clsHoodQuads/clsHoodHands used to initialize this
        % instance
        function pout = clsPtsHands(pin,handparams,objUID,camSet,DEBUG)
            pout@clsPtsFrenet(pin);
            if isa(pin,'clsPtsHands')
                for i=1:size(pin.hands,2)
                    pout.hands{i} = pin.hands{i};
                end
            end
            pout.handparams = handparams;
            if nargin < 5
                pout.DEBUG = 0;
            else
                pout.DEBUG = DEBUG;
            end
            pout.imageparams = [];
            pout.objUID = objUID;
            pout.camSet = camSet;
        end
        
        function hands = concatenate(hands,newhands)
            if ~isequal(hands.objUID, newhands.objUID) || ~isequal(hands.camSet,newhands.camSet)
                error('clsPtsHands.concatenate: cant concatenate unless objUID and camSet are identical');
            end
            hands = hands.concatenate@clsPtsFrenet(newhands);
            hands.handSetList = [hands.handSetList newhands.handSetList];
        end
        
        function pp = setImageParams(pp,imageparams)
            pp.imageparams = imageparams;
        end
        
        function hands = clearHandSetListHoods(hands)
            for i=1:hands.num()
                hands.handSetList{i} = hands.handSetList{i}.clearHood();
                hands.handSetList{i} = hands.handSetList{i}.clearOcclusionPts();
            end
        end
        
        % Import antipodal labels from elsewhere. Typically, we will
        % generate labels in clsLearning and then import them here.
        function hands = importAntipodalLabels(hands,handsList,labels)
            for i=1:size(handsList,1)
                pair = handsList(i,:);
                hands.handSetList{pair(1)}.antipodalFull(pair(2)) = labels(i);
            end
        end
        
        % Calculate <pos> in hand reference frame for each elt in handsList
        % input: pos -> 3x1 vector denoting pos in global frame
        % output: localPosList -> 3xn matrix denoting pose relative to
        %                         hands
        function localPosList = projectIntoHandFrames(hands,pos,handsList)
            if nargin < 3
                handsList = hands.getAllHands();
            end
            
            localPosList = zeros(3,size(handsList,1));
            for i=1:size(handsList,1)
                pair = handsList(i,:);
                hs = hands.handSetList{pair(1)};
                Hand = hs.Hands{pair(2)};
                localPosList(:,i) = Hand.F'*pos;
            end            
        end
        
        
        % Get local cam pos for each hand on list.
        % output: camPosLocalList -> 3xn matrix of cam poses
        function imageCamPosList = getImageCamPos(hands,handsList)
            if nargin < 2
                handsList = hands.getAllHands();
            end
            
            imageCamPosList = zeros(3,size(handsList,1));
            for i=1:size(handsList,1)
                pair = handsList(i,:);
                hs = hands.handSetList{pair(1)};
                Hand = hs.Hands{pair(2)};
                imageCamPosList(:,i) = hs.imageCamPos(:,pair(2));
            end
        end
        
        
        % Check whether hands are in collision with <cloud>
        % input: cloud -> clsPts
        % output: labelsCollision -> 1xn binary vector indicating whether
        % each hand in hands.getAllHands() is in collision wrt <cloud>.
        function labelsCollision = getHandsCollision(hands,cloud)
            
            list = hands.getAllHands();
            labelsCollision = zeros(1,size(list,1)); % init to NOT in collision

            % Set hands.cloud to <cloud>. Set local hoods in handSetList
            hands = hands.setCloud(clsPts(cloud));
            hands = hands.voxelizeCloud(0.003);
            maxDiameterClosingRegion = max([(hands.handparams.handOD - hands.handparams.fw) hands.handparams.handDepth hands.handparams.handHeight/2]);
            hands = hands.evalBall(maxDiameterClosingRegion);
            hands = hands.setHood();
            
            for i=1:size(list,1)
                
                % Get hand
                pair = list(i,:);
                hs = hands.handSetList{pair(1)};
                Hand = hs.Hands{pair(2)};
                
                % Check whether this hand is in collision
                croppedHoodPts = Hand.F'*(hs.hood.pts - repmat(Hand.sample,1,hs.hood.num()));
                idx = Hand.getPtsInFingers(croppedHoodPts);
                if size(idx,2) > 0
                    labelsCollision(i) = 1;

                    % if DEBUG, display all in-collision hands
                    if ismember('clsPtsHands.getHandsCollision',hs.DEBUG)
                        incollision = find(labelsCollision);
                        if size(incollision,2) > 0
                            pair = list(randi(size(incollision,2)),:);
                            figure;
                            Hand.plotHands(croppedHoodPts);
                        end
                    end
                    
                end
            end
            
        end
        

        % Calculate candidates and labels. Then balance results and clear
        % clouds.
        % input: ptsView -> cloud to use to calculate candidates
        %        ptsMesh -> mesh to use to calculate ground truth
        function hands = calcBalancedCandidatesAndLabels(hands,ptsView,ptsMesh,curvAxisProb)

            % Get grasp candidates
            hands = hands.getGraspCandidates(ptsView, curvAxisProb);

            % If no candidates were found, return
            if hands.num() == 0
                return;
            end
            
            % Get ground truth labels using full mesh
            hands = hands.calculateLabels(ptsMesh);
            [handsListPos, ~, handsListHalf] = hands.getAntipodalHandList();
            handsListNeg = setdiff(hands.getAllHands(),handsListPos,'rows');
            hands = hands.balance(handsListPos,handsListNeg);
            
            % Compress clsPtsHands
            hands = hands.clearCloud();
            hands = hands.clearHandSetListHoods();

        end

        
        % Get ground truth labels for hands in <this.handsView>.
        % input: ptsMesh -> clsPtsNormals describing a mesh for this object
        % PREREQUISITE: must have run getGraspCandidates first
%         function obj = getLabels(obj,ptsMesh)
        function hands = calculateLabels(hands,ptsMesh)

            if hands.num() == 0
                error('clsObjectCloud::getLabels: you need to call getGraspCandidates prior to running this function.');
            end

            ptsMesh = clsPtsHood(ptsMesh);
            ptsMesh = ptsMesh.calcNormalsClear();
%             figure; hold('on');
%             ptsMesh.plot('k');
%             ptsMesh.plotNormals(int32(ptsMesh.num()/2));
            hands = hands.setCloud(ptsMesh);

            % Get a large enough diameter around each point to enable us to
            % evaluate ground truth.
            maxDiameterClosingRegion = max([(hands.handparams.handOD - hands.handparams.fw) hands.handparams.handDepth hands.handparams.handHeight/2]);
            hands = hands.evalBall(maxDiameterClosingRegion);
            
            % Evaluate ground truth for hands in <handsMeshCloud>
            for i=1:hands.num()
                hs = hands.handSetList{i};
                hs = hs.setHood(hands.cloud.prune(hands.nhIdx{i})); % reset pts neighborhood with pts from mesh
                if hs.foundhand
                    hs = hs.labelHands();
                    hands.handSetList{i} = hs;
                end                
            end
            
            if ismember('clsPtsHands.calculateLabels',hs.DEBUG)
                figure;
                ptsMesh.plot(); hold on;
                [handsListPos, ~, ~] = hands.getAntipodalHandList();
                subList = handsListPos(randperm(size(handsListPos,1),min(10,size(handsListPos,1))),:); % get 10 random candidates
                hands.plotHandList(subList);
                lightangle(-45,30);
                title('10 randomly sampled positive grasps');
            end
        end
        
        function imageList = getImageList(hands,handsList)
            imageList = [];
            for i=1:size(handsList,1)
                pair = handsList(i,:);
                hs = hands.handSetList{pair(1)};
                imageList{i} = hs.images{pair(2)};
            end
        end
        
        % Get antipodal labels for hands in <handList>
        function labels = getAntipodalLabels(hands, handList)
            if nargin < 2
                handList = hands.getAllHands();
            end
            labels = [];
            for i=1:size(handList,1)
                pair = handList(i,:);
                if size(hands.handSetList{pair(1)}.antipodalFull,2) >= pair(2)
                    labels(i) = hands.handSetList{pair(1)}.antipodalFull(pair(2));
                else
                    labels(i) = 9; % if there are no antipodal labels, then just put a 9 in there.
                end
            end
        end
        
        % Get handLists for positive, negative, and half grasps.
        function [handsListPos, handsListNeg, handsListHalf] = getAntipodalHandList(hands)            
            [listPositives, listPositivesHalf] = hands.getAntipodal();
            if size(listPositives,1) > 0
                listNegatives = setdiff(hands.getAllHands(),listPositives,'rows');
            else
                listNegatives = hands.getAllHands();
            end
            handsListPos = listPositives;
            handsListNeg = listNegatives;
            handsListHalf = listPositivesHalf;
        end
        
        
        % Calculate images. Then, clear intermediate data.        
        function hands = calculateImagesClear(hands,thisView)

            hands = hands.calculateImages(thisView); % Populate image data

            % Compress clsPtsHands
            hands = hands.clearCloud();
            hands = hands.clearHandSetListHoods();
        end
        
        
        % Calculate images for caffe. Results go into 
        % input: handsList -> handsList describing hands for which to calculate
        % image.
        % The purpose of the flip is to enable us to train using the image
        % "both ways" about the x axis. Since this is physically symmetric,
        % choice of axis is arbitrary and we want to train it both ways.
        % input: 
        %        flip -> indicates whether to flip image about x axis. 
        %                0 = no flip; 1 = flip
        %        cloud -> pt cloud used to calculate images.        
         function hands = calculateImages(hands,cloud)
            
            hands = hands.setCloud(cloud);
            hands = hands.calcCloudNormals();
            maxDiameterImageRegion = max([hands.imageparams.imageOD hands.imageparams.imageDepth hands.imageparams.imageHeight/2]);
            hands = hands.evalBall(maxDiameterImageRegion);
            hands = hands.setHood();

            % Regular for loop
%             handsList = hands.getAllHands();
%             for ii=1:size(handsList,1)
%                 pair = handsList(ii,:);
%                 i = pair(1); j = pair(2);
%                 hands.handSetList{i} = hands.handSetList{i}.setImageParams(hands.imageparams);
%                 hands.handSetList{i} = hands.handSetList{i}.calculateImage(j);
%             end

            % parfor loop
            handsList = hands.getAllHands();
            handSetList = hands.handSetList;
            imageparams = hands.imageparams;
            parfor ii=1:hands.num()
                if ismember(ii, handsList(:,1))
                    handSetList{ii} = handSetList{ii}.setImageParams(imageparams);
                    jj = find(ii==handsList(:,1));
                    orientations = handsList(jj,2)';
                    for j = 1:size(orientations,2)
                        handSetList{ii} = handSetList{ii}.calculateImage(orientations(j));
                    end
                else
                    continue;
                end
            end
            hands.handSetList = handSetList;
            
            % If DEBUG, display randomly selected image from <list> paired
            % with raw points used to create the image.
            if ismember('clsPtsHands.calculateImage',hands.DEBUG)
                
                % display a randomly select elt from list
                if nargin < 3
                    handsList = hands.getAllHands();
                end
                pair = handsList(randi(size(handsList,1)),:);
                image = hands.handSetList{pair(1)}.images{pair(2)};
                
                % Display raw points
                hs = hands.handSetList{pair(1)};
                hood = hs.hood;
                Hand = hs.Hands{pair(2)};
                hood = hood.transform(Hand.F',Hand.sample);
                [~, croppedHoodPts] = getPts4UnitImage(Hand,hood.pts,hands.imageparams);
                figure;
                title('cropped/uncropped pts for random HandSet');                
                subplot(1,2,1); hood.plot(); title('uncropped');
                xlabel('x'); ylabel('y'); zlabel('z');
                subplot(1,2,2); plot3(croppedHoodPts(1,:),croppedHoodPts(2,:),croppedHoodPts(3,:),'m.'); title('cropped');
                xlabel('x'); ylabel('y'); zlabel('z');
                axis([0 1 0 1 0 1]);
                
                % Display resulting image
                figure;
                subplot(3,2,1); imshow(image(:,:,1:3));
                subplot(3,2,2); imshow(image(:,:,4));
                subplot(3,2,3); imshow(image(:,:,5:7));
                subplot(3,2,4); imshow(image(:,:,8));
                subplot(3,2,5); imshow(image(:,:,9:11));
                subplot(3,2,6); imshow(image(:,:,12));
                
            end
            
         end
        
        
        % Calculate number of hands contained in this class
        function num = numHands(hands)
            num = size(hands.getAllHands(),1);
        end
        
        
        % Set .hood parameter for all elts in .handSetList given
        % corresponding neighborhood in <hands>.
        % PREREQUISITE: must have run .getGraspCandidates first. OW,
        % .handSetList is empty.
        function hands = setHood(hands)
            for i=1:hands.num()
                hands.handSetList{i} = hands.handSetList{i}.setHood(hands.cloud.prune(hands.nhIdx{i}));
            end
        end
        
        
        % Sample grasp candidates from <cloud>. Seeds a local search at
        % each point in <hands>.
        % input: cloud -> cloud to be used for the search.
        % input: curveAxisProb -> probability of selecting curvature axis
        function hands = getGraspCandidates(hands,cloud,curveAxisProb)

            initBite = 0.015;
            dAmount = 0.005;
            
            % Set cloud and calculate normals
            hands = hands.setCloud(cloud);
            hands = hands.calcCloudNormals();
            if ismember('clsPtsHands.getGraspCandidates',hands.DEBUG) % plot hands.cloud and the corresponding normals
                figure;
                hands.cloud.plot(); hold on;
                hands.cloud.plotNormals(100);
                title('raw normals calculated for cloud')
            end

            % Prune pts w/ no neighbors
            hands = hands.evalBall(0.01);
            idxLarge = hands.getLargeNeighborhoods(20);
            hands = hands.prune(idxLarge);
            if hands.num() == 0
                return;
            end

            % Calculate local reference frame for each sample
            hands = hands.evalFrenetFrame();
            if ismember('clsPtsHands.getGraspCandidates',hands.DEBUG) % plot the samples with the frenet frame normals
                figure;
                hands.plot(); hold on;
                hands.plotNormals();
                title('normals calculated as part of Frenet frame');
            end

            % Voxelize hands.cloud to speed calculations
            hands = hands.voxelizeCloud(0.003);
            
            % Do a local search for hands around each sample
            maxDiameterClosingRegion = max([(hands.handparams.handOD - hands.handparams.fw) hands.handparams.handDepth hands.handparams.handHeight/2]);
            hands = hands.evalBall(maxDiameterClosingRegion);
            
            handSetList = {};
            parfor i=1:hands.num()
            %for i=1:hands.num()
                hs = clsHandSet(hands.handparams,hands.DEBUG);
                hs = hs.setSampleF(hands.pts(:,i),hands.F{i});
                hs = hs.setHood(hands.cloud.prune(hands.nhIdx{i}));
                hs = hs.populateHands(initBite,dAmount,curveAxisProb);
                if sum(hs.hands,2) > 0
                    handSetList{i} = hs;
                end
            end
            hands.handSetList = handSetList;
            hands = hands.prune(hands.getNonHands()); % compress structure by pruning non-hands

            if ismember('clsPtsHands.getGraspCandidates',hands.DEBUG)
                figure;
                cloud.plot(); hold on;
                handList = hands.getAllHands();
                subList = handList(randperm(size(handList,1),10),:); % get 10 random candidates
                hands.plotHandList(subList);
                lightangle(-45,30);
                title('10 randomly sampled grasp candidates');
            end
        end
        

        
        % Calculate indices of non-empty hands
        % output: nonemptyhands -> array of non-empty hand indices
        function nonemptyhands = getNonHands(hands)
            nonemptyhands = find(~cellfun(@isempty,hands.handSetList));
        end
        
        % Get handList of all hand hypotheses
        function handList = getAllHands(hands)
            if hands.num() > 0
                handListPartial{hands.num()} = [];
                for i=1:hands.num()
                    hs = hands.handSetList{i};
                    if hs.foundhand
                        orientations = find(hs.hands);
                        handListPartial{i} = [repmat(i,1,sum(hs.hands)); orientations]';
                    end
                end
                handList = cell2mat(handListPartial');
            else
                handList = [];
            end
        end

        % Get distances between bottom of grasp and the object surface for
        % all hands. This is used to measure whether the grasp is still
        % "good" after perspective ablation.
        % input: handList -> hand list for which to obtain
        %                       dist-from-bottom.
        % output dists -> column vector of distances.
        function dists = getSurfaceAboveBottom(hands, handList)
            dists = zeros(size(handList,1),1);
            for i=1:size(handList,1)
                pair = handList(i,:);
                fh = hands.handSetList{pair(1)}.fh{pair(2)};
                dists(i,1) = fh.surface - fh.bottom;
            end
        end
        
        % Calculate coordinate of lowest (smallest z value) fingertip.
        function lowestFingertip = getLowestFingertip(hands, handList)
            lowestFingertip = zeros(3,size(handList,1)); % preallocate
            for i=1:size(handList,1)
                pair = handList(i,:);
%                 [graspSurfaceBase, graspBottomBase, graspTopBase] = hands.handSetList{pair(1)}.fh{pair(2)}.getHandParameters();
                [graspSurfaceBase, graspBottomBase, graspTopBase, graspLeftTop, graspRightTop, graspLeftBottom, graspRightBottom] = hands.handSetList{pair(1)}.fh{pair(2)}.getHandParameters();
                if graspLeftTop(3) < graspRightTop(3)
                    lowestFingertip(:,i) = graspLeftTop;
                else
                    lowestFingertip(:,i) = graspRightTop;
                end
            end
        end
        
        
        % Get handList of antipodal hands. "Full antipodal" means that the
        % hand is definitely antipodal. "Half antipodal" means that it
        % would be antipodal under more relaxed conditions.
        % output: listFullAntipodal -> nx2 matrix of pairs. Each pair has
        %                               the form (hand, orientation).
        %         listHalfAntipodal -> 
        function [listFullAntipodal, listHalfAntipodal] = getAntipodal(ptsHands)
            if ptsHands.num() == 0
                listFullAntipodal = zeros(0,2);
                listHalfAntipodal = zeros(0,2);
            else
                listFullAntipodalPartial{ptsHands.num()} = [];
                listHalfAntipodalPartial{ptsHands.num()} = [];
                for i=1:ptsHands.num()
                    hs = ptsHands.handSetList{i};
                    numAntipodalFull = sum(hs.antipodalFull,2);
                    if numAntipodalFull > 0
                        orientations = find(hs.antipodalFull);
                        listFullAntipodalPartial{i} = [repmat(i,1,numAntipodalFull); orientations]';
                    end
                    numAntipodalHalf = sum(hs.antipodalHalf,2);
                    if numAntipodalHalf > 0
                        orientations = find(hs.antipodalHalf);
                        listHalfAntipodalPartial{i} = [repmat(i,1,numAntipodalHalf); orientations]';
                    end
                end
                listFullAntipodal = cell2mat(listFullAntipodalPartial');
                listHalfAntipodal = cell2mat(listHalfAntipodalPartial');
                if size(listFullAntipodal,1) == 0
                    listFullAntipodal = zeros(0,2);
                end
                if size(listHalfAntipodal,1) == 0
                    listHalfAntipodal = zeros(0,2);
                end
            end
        end

        
        % Subsample hands from <negatives> so that there are an equal
        % number of positives and negatives.
        % input: positives -> nx2 handList denoting positives
        %        negatives -> nx2 handList denoting negatives.
        function hands = balance(hands,positives,negatives)
            numPos = size(positives,1); numNeg = size(negatives,1);
            if numNeg >= numPos
                list = [positives; negatives(randperm(numNeg,numPos),:)];
            else
                list = [negatives; positives(randperm(numPos,numNeg),:)];
            end
            list = list(randperm(size(list,1)),:);
            hands = hands.pruneHandList(list);
        end

        
        % Plot hands in <handList>
        % input: handList -> nx2 matrix of pairs. Each pair has the form
        %                     (hand, orientation).
        function plotHandList(hands, handList, colr,plotArrows)
            if nargin < 4
                plotArrows = 0;
            end
            if nargin < 3
                colr = 'b';
            end
            for i=1:size(handList,1)
                hand = handList(i,1);
                orientation = handList(i,2);
                if isobject(hands.handSetList{hand})
                    hands.handSetList{hand}.plotHand(colr,orientation,plotArrows); % display hands in blue
                end
            end
        end
        
        % prune elts
        % input: elts2keep -> indices of elts to keep (prune rest)
        function nhout = prune(nh,elts2keep)
            nhout = nh.prune@clsPtsFrenet(elts2keep);
            nhout.handSetList = cell(1,size(elts2keep,2));
            for i=1:size(elts2keep,2)
                if size(nh.handSetList,2) >= elts2keep(i)
                    nhout.handSetList{i} = nh.handSetList{elts2keep(i)};
                end
            end
        end
        
        % Prune all hands not on <handList2keep>.
        % input: handList2keep -> handList to keep. Prune all other hands.
        function hh = pruneHandList(hh,handList2keep)
            handsets = unique(handList2keep(:,1))'; % unique handsets in handList2keep
            hhtemp = hh.prune(handsets); % prune handsets
            for i=1:size(handsets,2)
                orientations = handList2keep(handList2keep(:,1) == handsets(i), 2); % orientations for this handset to keep
                hhtemp.handSetList{i} = hhtemp.handSetList{i}.prune(orientations');
            end
            hh = hhtemp;
        end
        


        
    end
    
end
