classdef clsHand
    
    properties
        
        % properties of the hand
        handparams;
        
        F; % this hand in base frame:  [q.normal cross(q.axis,q.normal) q.axis];
        sample;
        
        % Calculated in evalFingers
        fspacing; % spacing of 2xn fingers
        fingers; % 1x(2n) binary vector of fingers that fit into cloud.
        
        % Calculated in evalHand
        hand; % 1xn binary vector of hands that fit the cloud. Calculated in evalHand
        
        % Various geometric parameters of the hand. Set in evalHand and
        % deepenHand.
        top; % (scalar)
        bottom; % (scalar)
        surface; % (scalar)
        left; % (scalar) horizontal position of inside of left finger
        right; % (scalar) horizontal position of inside of right finger
        center; % (scalar) horizontal position of center of hand.
        width; % (scalar) width of object (i.e. the pts) between fingers
        
    end
    
    methods
        
        function hand = clsHand(F,sample,handparams)
            hand.F = F;
            hand.sample = sample;
            n = 10;        % number of finger placements to consider over a single hand diameter.
            fshalf = linspace(0,handparams.handOD-handparams.fw,n);
            hand.fspacing = [(fshalf - handparams.handOD + handparams.fw) fshalf];
            hand.fingers = zeros(1,2*n);
            hand.handparams = handparams;
            hand.width = 0;
        end

        % Transform clsPtsXXX according to reference frame of <this> hand
        % input: p -> instance of clsPtsXXX
        % output: pout -> instance of clsPtsXXX
        function pout = transform(h,p)
            pout = p.transform(h.F',h.sample);
        end
        
        
        % Binary output that denotes whether hand.hand contains any hands.
        % PREREQUISITE: you should run evalHand, evalFingers first to
        % populate hand.fingers and hand.hand
        function found = foundhand(hand)
            found = sum(hand.hand,2) > 0;
        end
        
        
        % Erode hand. If there are multiple viable hands, delete everything
        % except for the center hand. Delete corresponding fingers as well.
        function hand = erodeHand(hand)

            handidx = find(hand.hand); % all hands
            if find(handidx,2) > 0                
                handerodedidx = handidx(1,ceil(size(handidx,2)/2)); % calculate middle hand
                handeroded = zeros(size(hand.hand));
                handeroded(1,handerodedidx) = 1; % delete everything except for middle hand
                hand.hand = handeroded;
                hand.fingers = [handeroded handeroded]; % reconstitute fingers for this middle hand
            end
            
        end
        
        
        % Calculate indices of pts that are contained inside of one of the
        % two fingers.
        function idx = getPtsInFingers(hand,pts)
            if ~hand.foundhand
                error('clsHand::getPtsInFingers: you must call this function for a found hand');
            end
            
            idx = find((pts(1,:) > hand.bottom) & (pts(1,:) < hand.top) ...
                & (pts(3,:) > -hand.handparams.handHeight) & (pts(3,:) < hand.handparams.handHeight) ...
                & (((pts(2,:) <= hand.left) & (pts(2,:) >= hand.left - hand.handparams.fw)) | ((pts(2,:) >= hand.right) & (pts(2,:) <= hand.right + hand.handparams.fw))));
        end
        

        % Calculate indices of pts that are contained within box of this
        % hand, i.e. between the fingers.
        function idx = getPtsBetweenFingers(hand,pts)
            if ~hand.foundhand
                error('clsHand::getPtsInBox: you must call this function for a found hand');
            end
            
            idx = find((pts(2,:) > hand.left) & (pts(2,:) < hand.right) ...
                & (pts(1,:) > hand.bottom) & (pts(1,:) < hand.top) ...
                & (pts(3,:) > -hand.handparams.handHeight) & (pts(3,:) < hand.handparams.handHeight));
        end
        
        % Calculate indices of pts that are contained within the specified
        % box.
        % input: pts -> 
        %        imageparams -> 
        function [idx, ptsout] = getPts4UnitImage(hand,pts,imageparams)
            if ~hand.foundhand
                error('clsHand::getPtsInBox: you must call this function for a found hand');
            end
            
            idx = find((pts(2,:) > hand.center - imageparams.imageOD/2) & (pts(2,:) < hand.center + imageparams.imageOD/2) ...
                & (pts(1,:) > hand.bottom) & (pts(1,:) < hand.bottom + imageparams.imageDepth) ...
                & (pts(3,:) > -imageparams.imageHeight) & (pts(3,:) < imageparams.imageHeight));
            
            ptsout = pts(:,idx);
            ptsout(1,:) = (ptsout(1,:) - hand.bottom) / imageparams.imageDepth;
            ptsout(2,:) = (ptsout(2,:) - (hand.center - imageparams.imageOD/2)) / imageparams.imageOD;
            ptsout(3,:) = (ptsout(3,:) + imageparams.imageHeight) / (2*imageparams.imageHeight);            
        end
        
        % output: .hand -> positions of all found hands
        %         .foundhand -> binary indicating hand found
        function hand = evalHand(hand)
            n = size(hand.fingers,2)/2; % by design, n should be integer
            hand.hand = bitand(hand.fingers(1,1:n),hand.fingers(1,n+1:end));
        end
        
        
        % input: pts ->
        %        fingerTemplate -> (optional) Tells the function to only
        %                           look for fingers at certain
        %                           positions. The default is to look
        %                           in all positions.
        % output: .fingers
        % PREREQUISITE: must have run setTopBottom prior to calling this
        %                function.
        function hand = evalFingersHands(hand,pts,fingerTemplate)
            
            % Default is to check all finger positions
            if nargin < 3
                fingerTemplate = ones(1,size(hand.fspacing,2));
            end
            
            % Initialize with zero fingers, hands found
            hand.fingers = zeros(size(hand.fingers));
            hand.hand = zeros(size(hand.hand));

            % Bail if there are points behind base of hand.
            if sum(pts(1,:) < hand.bottom) > 0
                return;
            end
            
            ptsCropped = pts(:,pts(1,:) < hand.top);

            % Bail if there are no points anywhere
            if size(ptsCropped,2) == 0
                return;
            end
            
            % Find free finger positions
            m = size(hand.fspacing,2);
            for j=1:m
                if fingerTemplate(1,j)
                    itemsInGap = (ptsCropped(2,:) > hand.fspacing(1,j)) & (ptsCropped(2,:) < (hand.fspacing(1,j) + hand.handparams.fw));
                    if sum(itemsInGap) == 0
                        hand.fingers(1,j) = 1;
                    end
                end
            end
            
            % Fin free hand positions
            hand = hand.evalHand();
            
        end
        
        
        % Select center hand and try to extend it into the object as deeply
        % as possible. Populate output hand parameters.
        % Must have run this.setF() and this.setPts() prior to calling this function.
        % input: initBite -> initial deepness
        %        maxamount -> amount by which to deepen
        %        normals -> (optional) If included, normal information will
        %        be included in clsFingerHand for the purpose of computing
        %        a ground truth antipodal grasp. Not needed if ground
        %        truth will not be calculated.
        function hand = deepenHand(hand,initBite, maxdeepenamount, damount, pts)

            if ~hand.foundhand
                error('called clsHand::deepenHand, but no hand was found.');
            end

            hand = hand.erodeHand(); % choose a single hand to push forward
            erodedHand = hand.hand;
            erodedFingers = hand.fingers;
            lastHand = hand;
            for amount = initBite+damount:damount:maxdeepenamount
                hand = hand.setTopBottom(amount);
                hand = hand.evalFingersHands(pts, erodedFingers); % set fingerTemplate = erodedFingers so that we only check for the eroded hand -- not all the hands.
                if ~hand.foundhand
                    break;
                end
                hand.hand = erodedHand; % make sure we continue to try to deepen the initially eroded hand
                hand.fingers = erodedFingers; % make sure we continue to try to deepen the initially eroded hand
                lastHand = hand;
            end
            
            hand = lastHand; % recover most deep hand
            
        end
        
        
        % Top is the x-position of the tips of the fingers. The equivalent
        % of <bite> for this hand. Bottom is set automatically based on
        % handDepth.
        function hand = setTopBottom(hand,top)
            hand.top = top;
            hand.bottom = top - hand.handparams.handDepth;
        end


        % Left and right are the inner surfaces of the two fingers. Set
        % based on the contents of <h.hand> and <h.fspacing>
        % output: .left, .right, .center -> undefined if hand.hand is all
        % zeros
        function hand = setLeftRight(hand)
            if sum(hand.hand) > 0
                idx = find(hand.hand);
                hand.left = hand.fspacing(1,idx) + hand.handparams.fw;
                hand.right = hand.fspacing(1,size(hand.hand,2) + idx);
                hand.center = 0.5 * (hand.left + hand.right);            
            end
        end

        % Calculate width of object between the fingers
        function h = setWidth(h,pts)
            idx = h.getPtsBetweenFingers(pts);
            if isempty(idx), h.width = 0; return; end
            ptsBetweenFingers = pts(:,idx);
            h.width = max(ptsBetweenFingers(2,:)) - min(ptsBetweenFingers(2,:));
        end
        
        % Set the gripper surface parameter. This is the point between the
        % fingers that is closest to the top of the gripper.
        function h = setSurface(h,pts)            
            if size(pts,2) == 0
                h.surface = h.top;
            else
                h.surface = min(pts(1,:));
            end            
        end
        
%         
%         function imgOcclusionDialated = getOccImages(Hand,imageparams,ptsOcclusionUnit)
% 
%             cellSize = 1 / imageparams.imageSize;
%             vertCells = min(floor(ptsOcclusionUnit(1,:) / cellSize) + 1, imageparams.imageSize);
%             horCells = min(floor(ptsOcclusionUnit(2,:) / cellSize) + 1, imageparams.imageSize);
%             idx = horCells + ((vertCells-1)*imageparams.imageSize);
%             acc = accumarray(idx',ptsOcclusionUnit(3,:)',[imageparams.imageSize*imageparams.imageSize 1]);
%             counts = accumarray(idx',1,[imageparams.imageSize*imageparams.imageSize 1]);
%             accavg = zeros(imageparams.imageSize*imageparams.imageSize,1);
%             accavg(idx) = acc(idx) .* (1./counts(idx));
%             accavg(idx) = max(accavg(idx)) - accavg(idx); % reverse depth so that closest points have largest value
%             imgOcclusion=flipud(reshape(accavg,imageparams.imageSize,imageparams.imageSize)');
%             % mg speedup c
%             %se = strel('square',3); % Used to dilate image. I suspect that this step helps generalize
%             %imgOcclusionDialated = imdilate(imgOcclusion,se);
%             %imgOcclusionDialated = im2uint8(mat2gray(imgOcclusionDialated));
%             % mg speedup u
%             imgOcclusionDialated = im2uint8(mat2gray(imgOcclusion));
%         end
            
        function [imgRGBDialated, imgDepthDialated] = getPtsImages(Hand,imageparams,ptsUnit,normalsUnit)
        
            cellSize = 1 / imageparams.imageSize;
            vertCells = min(floor(ptsUnit(1,:) / cellSize) + 1, imageparams.imageSize);
            horCells = min(floor(ptsUnit(2,:) / cellSize) + 1, imageparams.imageSize);
            idx = horCells + ((vertCells-1)*imageparams.imageSize);

            % Calculate RGB image
            accrgb(1,:) = accumarray(idx',normalsUnit(1,:)',[imageparams.imageSize*imageparams.imageSize 1])';
            accrgb(2,:) = accumarray(idx',normalsUnit(2,:)',[imageparams.imageSize*imageparams.imageSize 1])';
            accrgb(3,:) = accumarray(idx',normalsUnit(3,:)',[imageparams.imageSize*imageparams.imageSize 1])';
            accrgbmag = sqrt(sum(accrgb.^2,1));
            nonzeros = find(accrgbmag);
            accrgb(:,nonzeros) = abs(accrgb(:,nonzeros) .* repmat(1./accrgbmag(1,nonzeros),3,1));
            imgRGB(:,:,1) = flipud(reshape(accrgb(1,:),imageparams.imageSize,imageparams.imageSize)');
            imgRGB(:,:,2) = flipud(reshape(accrgb(2,:),imageparams.imageSize,imageparams.imageSize)');
            imgRGB(:,:,3) = flipud(reshape(accrgb(3,:),imageparams.imageSize,imageparams.imageSize)');

            % Calculate depth image
            acc = accumarray(idx',ptsUnit(3,:)',[imageparams.imageSize*imageparams.imageSize 1]);
            counts = accumarray(idx',1,[imageparams.imageSize*imageparams.imageSize 1]);
            accavg = zeros(imageparams.imageSize*imageparams.imageSize,1);
            accavg(idx) = acc(idx) .* (1./counts(idx));
            accavg(idx) = 1 - accavg(idx); % reverse depth so that closest points have largest value
            imgDepth = flipud(reshape(accavg,imageparams.imageSize,imageparams.imageSize)');
            
            % mg speedup c
            se = strel('square',3); % Used to dilate image. I suspect that this step helps generalize            
            imgRGBDialated = imdilate(imgRGB,se);
            imgRGBDialated = im2uint8(mat2gray(imgRGBDialated));
            imgDepthDialated = imdilate(imgDepth,se);
            imgDepthDialated = im2uint8(mat2gray(imgDepthDialated));
            % mg speedup u
%             imgRGBDialated = im2uint8(mat2gray(imgRGB));
%             imgDepthDialated = im2uint8(mat2gray(imgDepth));
            
        end
    
        
        % Calculate position of the hand in base frame.
        % Must be called after this.deepenHand()
        % ouput: graspSurfaceBase -> position of hand middle on object
        %                           surface
        %        graspBottomBase -> position of hand middle closest to
        %                           actuator
        %        graspTopBase -> position of hand middle between fingertips
        function [graspSurfaceBase, graspBottomBase, graspTopBase, graspLeftTop, graspRightTop, graspLeftBottom, graspRightBottom] = getHandParameters(hand)
            
            % Convert gripper parameters into base frame
            graspSurfaceBase = hand.F*[hand.surface; hand.center; 0] + hand.sample;
            graspBottomBase = hand.F*[hand.bottom; hand.center; 0] + hand.sample;
            graspTopBase = hand.F*[hand.top; hand.center; 0] + hand.sample;
            graspLeftTop = hand.F*[hand.top; hand.left; 0] + hand.sample;
            graspRightTop = hand.F*[hand.top; hand.right; 0] + hand.sample;
            graspLeftBottom = hand.F*[hand.bottom; hand.left; 0] + hand.sample;
            graspRightBottom = hand.F*[hand.bottom; hand.right; 0] + hand.sample;
            
        end        

        function plotFingers(g,pts)
            plot(pts(2,:),pts(1,:),'bx');
            hold on;
            nonzerogaps = find(g.fingers);
            bite = g.handparams.handDepth + g.bottom;
            for i=1:size(nonzerogaps,2)
                j = nonzerogaps(i);
                plotlines = [g.fspacing(j) g.fspacing(j) g.fspacing(j)+g.handparams.fw g.fspacing(j)+g.handparams.fw g.fspacing(j); bite g.bottom g.bottom bite bite];
                plot(plotlines(1,:),plotlines(2,:),'c'); % ,'Color',[c c c]);
            end
            axis equal;
        end

        function plotHands(g,pts)
            plot(pts(2,:),pts(1,:),'bx');
            hold on;
            nonzerogaps = find(g.hand);
            bite = g.handparams.handDepth + g.bottom;
            for i=1:size(nonzerogaps,2)
                j = nonzerogaps(i);
                plotlines = [g.fspacing(j) g.fspacing(j) g.fspacing(j)+g.handparams.fw g.fspacing(j)+g.handparams.fw g.fspacing(j); bite g.bottom g.bottom bite bite];
                plot(plotlines(1,:),plotlines(2,:),'m'); % ,'Color',[c c c]);
                plotlines(1,:) = plotlines(1,:) + g.handparams.handOD - g.handparams.fw;
                plot(plotlines(1,:),plotlines(2,:),'m');
            end
%             plot(g.center, g.surface,'mx'); % plot a single magenta point on grasp surface
            axis equal;
        end

        
        
    end
    
end
