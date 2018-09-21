classdef clsHandSet
    
    properties
        
        DEBUG;
        
        t; % 1xn vector of hand orientations (n=8 in our applications)

        handparams; % properties of the hand        
        imageparams; % properties of image
        
        % The following variables must be populated prior to executing
        % .populateHands
        hood; % points in local neighborhood (clsPtsNormals), set in .setHood
        sample; % center point, set in .setSampleF
        F; % coordinate frame of quadric: [q.normal cross(q.axis,q.normal) q.axis]; , set in .setSampleF
        
        % Calculated in populateHands. Grasp candidates in |t| orientations
        % are encoded as .Hands as clsHand elts.
        Hands; % 1 x |t| cell array of clsHand elts
        hands; % 1 x |t| binary vector indicating the presence of a hand in each orientation
        
        % Calculated in labelHands. Denotes whether each hand is a good
        % grasp or not.
        antipodalFull; % 1 x |t| binary vector indicating whether one side of grasp is antipodal
        antipodalHalf; % 1 x |t| binary vector indicating whether both sides of grasp are antipodal

        regionLabel; % 1 x |t| binary vector indicating whether grasp is in a desired region or not
        
        % Occluded regions relative to pts in hood. Calculated in calculateOcclusion. 
        occlusionPts;
        
        % image for each elt in handset
        images;
        imageCamPos; % 3 x |t| matrix of cam positions at time images was calculated
        
    end
    
    methods
        

        function hs = clsHandSet(handparams,DEBUG)
            if nargin < 2
                hs.DEBUG = 0;
            else
                hs.DEBUG = DEBUG;
            end
            nt = 8; % num orientations
%             hs.t = linspace(-pi,pi,nt+1);
            hs.t = linspace(-pi/2,pi/2,nt+1);
%             hs.t = linspace(-pi/4,pi/4,nt+1);
            hs.t = hs.t(1,1:nt);
%             hs.t = hs.t + rand*pi/nt; % add a small random offset
%             hs.t = hs.t + pi;
            hs.handparams = handparams;
        end
        
        function hs = clearHood(hs)
            hs.hood = [];
        end
        
        function hs = clearOcclusionPts(hs)
            hs.occlusionPts = [];
        end
        
        % Prune orientations not in <orientations2keep>
        function hs = prune(hs, orientations2keep)
            
            % delete orientations from .hands vector
            hs.hands = zeros(1,size(hs.t,2));
            hs.hands(1,orientations2keep) = 1;
            
            hs.antipodalFull = [hs.antipodalFull zeros(1,size(hs.hands,2) - size(hs.antipodalFull,2))] & hs.hands;
            hs.antipodalHalf = [hs.antipodalHalf zeros(1,size(hs.hands,2) - size(hs.antipodalHalf,2))] & hs.hands;
            
            Hands = cell(1,size(hs.t,2));
            for i=1:size(orientations2keep,2)
                Hands{orientations2keep(i)} = hs.Hands{orientations2keep(i)};
            end
            hs.Hands = Hands;
        end

        function hs = setImageParams(hs,imageparams)
            hs.imageparams = imageparams;
        end
        
        function found = foundhand(hs)
            found = sum(hs.hands,2) > 0;
        end
        
        function hs = setSampleF(hs, sample, F)
            hs.sample = sample;
            hs.F = F;
        end
        
        function hs = setHood(hs, hood)
            hs.hood = hood;
        end
        
        function pp = pts(hs)
            pp = hs.hood.pts;
        end

        function pp = ptsCamSource(hs)
            pp = hs.hood.ptsCamSource;
        end

        function pp = camPos(hs)
            pp = hs.hood.camPos;
        end

        function pp = normals(hs)
            pp = hs.hood.normals;
        end


        % Calculate 3x3 rotation matrix for <theta>. This function contains
        % the formula for transforming an elt of hs.t to a rotation matrix.
        function R = getRotation(hs,theta,curveAxisProb)
%             RR = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]; % rotate about curvature axis
%             RR = [1 0 0; 0 cos(theta) -sin(theta); 0 sin(theta) cos(theta)]; % rotate about normal axis
%             RR = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)]; % rotate about binormal axis

            
            % sample randomly between conventional and mug lip grasp (-mg)
            if rand<curveAxisProb
                % conventional
                RR = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1];
            else
                % mug lip grasp
                RR1 = [1 0 0; 0 cos(pi/2) -sin(pi/2); 0 sin(pi/2) cos(pi/2)]; % rotate about normal axis
                RR2 = [cos(theta+pi/2) -sin(theta+pi/2) 0; sin(theta+pi/2) cos(theta+pi/2) 0; 0 0 1];
                RR = RR1*RR2;          

                % Sample a random orientation
%                 n = 1000;
%                 inCone = [];
%                 while size(inCone,2) == 0
%                     samp = randn(4,n);
%                     sampleMag = sqrt(sum(samp.^2,1));
%                     sampleUnit = samp ./ repmat(sampleMag,4,1);
%                     inCone = find([1 0 0 0]*sampleUnit > cos(pi/2)); % inside a 90 deg arc about zero rotation
%                 end
%                 sampleQuat = sampleUnit(:,inCone(1));
%                 RR = quat2rotm(sampleQuat');
            end

            R = hs.F*[-1 0 0; 0 1 0; 0 0 -1]*RR; 
        end
         
        
        % Perform a local search for each orientation in this clsHandSet.
        % For each found hand, populate .hands, .Hands, .left, .right,
        % .surface.
        % PREREQUISITE: must have already populated .sample, .F, .hood
        % output: .hands, .Hands, .left, .right, .surface
        function hs = populateHands(hs,initBite,damount,curveAxisProb)

            hs.hands = zeros(1,size(hs.t,2));
            
            for i = 1:size(hs.t,2)

                % get rotation matrix for this configuration
                FHand = hs.getRotation(hs.t(i),curveAxisProb);
                Hand = clsHand(FHand,hs.sample,hs.handparams);

                % transform pts into rotated local Frenet frame
                croppedHoodPts = Hand.F'*(hs.hood.pts - repmat(Hand.sample,1,hs.hood.num()));
                
                % crop on hand height
                idx = (croppedHoodPts(3,:) > -hs.handparams.handHeight) & (croppedHoodPts(3,:) < hs.handparams.handHeight); % crop on hand height
                
                Hand = Hand.setTopBottom(initBite);
                Hand = Hand.evalFingersHands(croppedHoodPts(:,idx));
                
                if Hand.foundhand()

                    if ismember('clsHandSet',hs.DEBUG)
                        figure;
                        Hand.plotFingers(croppedHoodPts(:,idx)); hold on;
                        Hand.plotHands(croppedHoodPts(:,idx));
                        title('fingers/hands prior to deepen');
                    end

                    Hand = Hand.deepenHand(initBite, Hand.handparams.handDepth, damount, croppedHoodPts(:,idx)); % Find deepest hand
                    
                    if ismember('clsHandSet',hs.DEBUG)
                        figure;
                        Hand.plotFingers(croppedHoodPts(:,idx)); hold on;
                        Hand.plotHands(croppedHoodPts(:,idx));
                        title('fingers/hands after deepen');
                    end
                                                        
                    Hand = Hand.setLeftRight();
                    Hand = Hand.setSurface(croppedHoodPts(:,idx));
                    Hand = Hand.setWidth(croppedHoodPts(:,idx));
                    
                    hs.hands(i) = 1;
                    hs.Hands{i} = Hand;
                end

            end
        end

        
        function hs = labelHands(hs)
            
            handidx = find(hs.hands);
            for j=1:size(handidx,2)
                
                k = handidx(j);
                Hand = hs.Hands{k};
                
                % transform pts into rotated local Frenet frame
                croppedHoodPts = Hand.F'*(hs.hood.pts - repmat(Hand.sample,1,hs.hood.num()));
                croppedHoodNormals = Hand.F'*hs.hood.normals;
                
                if ismember('clsHandSet',hs.DEBUG)
                    figure;
                    Hand.plotHands(croppedHoodPts);
                end
                
                % if fingers collide with points in mesh, discard hands
                idx = Hand.getPtsInFingers(croppedHoodPts);
                if size(idx,2) > 0
                    
                    fullGrasp = 0;
                    halfGrasp = 0;
                    
                    if ismember('clsHandSet',hs.DEBUG)
                        figure;
                        Hand.plotHands(croppedHoodPts(:,idx));
                    end
                
                else
                
                    idx = Hand.getPtsBetweenFingers(croppedHoodPts);
                    croppedHoodPts = croppedHoodPts(:,idx);
                    croppedHoodNormals = croppedHoodNormals(:,idx);

                    if ismember('clsHandSet',hs.DEBUG)
                        figure;
                        Hand.plotHands(croppedHoodPts);
                    end

                    a = clsAntipodal(croppedHoodPts,croppedHoodNormals);
                    [fullGrasp, halfGrasp] = a.evalAntipodal(0.003); % extremalThreshold

                end
                
                hs.antipodalFull(1,k) = fullGrasp;
                hs.antipodalHalf(1,k) = halfGrasp;
                
            end
        end                    
        
        
        function plotPts(p)
            plot3(p.pts(1,:), p.pts(2,:), p.pts(3,:), '.b', 'MarkerSize', 2);
            xlabel('x');
            ylabel('y');
            zlabel('z');
            hold on;
            plot3(p.sample(1),p.sample(2),p.sample(3),'rx','LineWidth',2); % sample located at origin
            axis equal;
        end
        
        
        function plotPtsNormals(p)
            plot3(p.pts(1,:), p.pts(2,:), p.pts(3,:), '.b', 'MarkerSize', 2);
            hold on;
            d = 0.02;
            plot3([p.pts(1,:); p.pts(1,:) + d*p.normals(1,:)], [p.pts(2,:); p.pts(2,:) + d*p.normals(2,:)], [p.pts(3,:); p.pts(3,:) + d*p.normals(3,:)]);            
            xlabel('x');
            ylabel('y');
            zlabel('z');
            hold on;
            plot3(p.sample(1),p.sample(2),p.sample(3),'rx','LineWidth',2); % sample located at origin
            axis equal;
        end
        
        % Plot one or more hands contained in this HandHyp class.
        % input: orientations -> variable length vector indicating which
        %                       orientations to display. Use one integer 
        %                       (between 1 and 8) per orientation
        %                       (OPTIONAL)
        %        colr -> color to use when displaying hands
        %        plotArrows -> binary: plot arrows pointing from orthogonal
        %                       directions
        function plotHand(hs,colr,orientations,plotArrows)
            if (nargin < 4)
                plotArrows = 0;
            end
            if (nargin < 3)
                orientations = find(hs.hands);
            end
            if (nargin < 2)
                colr = 'b';
            end
            
            for i=1:size(orientations,2)
                j = orientations(i);
                Hand = hs.Hands{j};
                
                [center, bottom, top] = Hand.getHandParameters();
                approach = Hand.F(:,1);
                ax = Hand.F(:,3);
                binormal = Hand.F(:,2);
                

                
                % These widths are easier to see (I use them for
                % presentations)
%                 hh.plotHandGeometry(fh.handparams.handHeight/2,fh.handparams.fw/2,fh.handparams.handOD,fh.handparams.handDepth,0.01,0.05,bottom,ax,approach); % fh,handparams.fw,od,fl,bl,stemHeight,position,ax,approach

                % these widths are technically correct
%                 hs.plotHandGeometry(Hand.handparams.handHeight,Hand.handparams.fw,Hand.handparams.handOD,Hand.handparams.handDepth,0.01,0.05,bottom,ax,approach,plotArrows); % fh,handparams.fw,od,fl,bl,stemHeight,position,ax,approach
                hs.plotHandGeometry(0.01,0.05,bottom,ax,approach,plotArrows); % fh,handparams.fw,od,fl,bl,stemHeight,position,ax,approach

                axis equal;

            end
        end
        
        % Plot hand given shape and pose parameters.
        % input: fh -> finger height
        %        handparams.fw -> finger width
        %        od -> outer diameter
        %        fl -> finger length (handDepth)
        %        bl -> base length
        %        stepHeight -> height of stem
        %        position, ax, approach -> position and orientation of hand.
        %        plotArrows -> binary: plot arrows pointing from orthogonal
        %                       directions
%         function plotHandGeometry(hs,fh,handparams,fw,od,fl,bl,stemHeight,position,ax,approach,plotArrows)
        function plotHandGeometry(hs,bl,stemHeight,position,ax,approach,plotArrows)

            R_b_h = [-cross(ax,approach) ax approach];

            fingerRightBaseRoot = [hs.handparams.handOD/2  hs.handparams.handOD/2 (hs.handparams.handOD/2 - hs.handparams.fw)  (hs.handparams.handOD/2 - hs.handparams.fw); ...
                               hs.handparams.handHeight/2 -hs.handparams.handHeight/2 -hs.handparams.handHeight/2         hs.handparams.handHeight/2; ...
                               0     0     0            0];
            fingerRightTop = (fingerRightBaseRoot + repmat([0;0;hs.handparams.handDepth],1,4));
            fingerRightBase = R_b_h * fingerRightBaseRoot + repmat(position,1,4);
            fingerRightTop = R_b_h * fingerRightTop + repmat(position,1,4);

            fingerLeftBase = (fingerRightBaseRoot .* repmat([-1;1;1],1,4));
            fingerLeftTop = (fingerLeftBase + repmat([0;0;hs.handparams.handDepth],1,4));
            fingerLeftBase = R_b_h * fingerLeftBase + repmat(position,1,4);
            fingerLeftTop = R_b_h * fingerLeftTop + repmat(position,1,4);

            baseLeft = [-hs.handparams.handOD/2  -hs.handparams.handOD/2  -hs.handparams.handOD/2  -hs.handparams.handOD/2; ...
                        hs.handparams.handHeight/2 -hs.handparams.handHeight/2 -hs.handparams.handHeight/2  hs.handparams.handHeight/2; ...
                        0     0    -bl   -bl];
            baseRight = (baseLeft .* repmat([-1;1;1],1,4));
            baseLeft = R_b_h * baseLeft + repmat(position,1,4);
            baseRight = R_b_h * baseRight + repmat(position,1,4);

            stemBase = [-hs.handparams.handHeight/2  -hs.handparams.handHeight/2  hs.handparams.handHeight/2  hs.handparams.handHeight/2; ...
                         hs.handparams.handHeight/2  -hs.handparams.handHeight/2 -hs.handparams.handHeight/2  hs.handparams.handHeight/2; ...
                        -bl    -bl   -bl   -bl];
            stemTop = (stemBase + repmat([0;0;-stemHeight],1,4));
            stemBase = R_b_h * stemBase + repmat(position,1,4);
            stemTop = R_b_h * stemTop + repmat(position,1,4);

            arrowZ = R_b_h * [0 0;0 0; -0.07 -0.14]  + repmat(position,1,2);
            arrowY = R_b_h * [0 0;0.07 0.14; hs.handparams.handDepth/2 hs.handparams.handDepth/2]  + repmat(position,1,2);
            arrowX = R_b_h * [0.07 0.14; 0 0; hs.handparams.handDepth/2 hs.handparams.handDepth/2]  + repmat(position,1,2);
            
            colr = 'y';
%             colr = [0.5 0.5 0.5];
            hs.plotRectangle3D(fingerRightBase,fingerRightTop,colr);
            hs.plotRectangle3D(fingerLeftBase,fingerLeftTop,colr);
            hs.plotRectangle3D(baseLeft,baseRight,colr);
            hs.plotRectangle3D(stemBase,stemTop,colr);

            if plotArrows
                quiver3(arrowZ(1,2),arrowZ(2,2),arrowZ(3,2),arrowZ(1,1) - arrowZ(1,2),arrowZ(2,1) - arrowZ(2,2),arrowZ(3,1) - arrowZ(3,2),'LineWidth',2,'MaxHeadSize',2,'Color','r')
                quiver3(arrowY(1,2),arrowY(2,2),arrowY(3,2),arrowY(1,1) - arrowY(1,2),arrowY(2,1) - arrowY(2,2),arrowY(3,1) - arrowY(3,2),'LineWidth',2,'MaxHeadSize',2,'Color','g')
                quiver3(arrowX(1,2),arrowX(2,2),arrowX(3,2),arrowX(1,1) - arrowX(1,2),arrowX(2,1) - arrowX(2,2),arrowX(3,1) - arrowX(3,2),'LineWidth',2,'MaxHeadSize',2,'Color','b')
            end
            
%             quiver3(arrowY(1,:),arrowY(2,:),arrowY(3,:),'k');
%             quiver3(arrowX(1,:),arrowX(2,:),arrowX(3,:),'k');
            
%             lightangle(-45,30);

        end

        % input: bottom -> 3x4 matrix of points describing a rectangle on "bottom"
        %                   of rectangle
        %        top -> 3x4 matrix of points describing a rectangle on "top"
        %                   of rectangle
        function plotRectangle3D(hs,bottom,top,colr)

            Mx = [bottom(1,:)' top(1,:)' [bottom(1,1:2) top(1,2:-1:1)]' [bottom(1,2:3) top(1,3:-1:2)]' [bottom(1,3:4) top(1,4:-1:3)]' [bottom(1,[4 1]) top(1,[1 4])]'];
            My = [bottom(2,:)' top(2,:)' [bottom(2,1:2) top(2,2:-1:1)]' [bottom(2,2:3) top(2,3:-1:2)]' [bottom(2,3:4) top(2,4:-1:3)]' [bottom(2,[4 1]) top(2,[1 4])]'];
            Mz = [bottom(3,:)' top(3,:)' [bottom(3,1:2) top(3,2:-1:1)]' [bottom(3,2:3) top(3,3:-1:2)]' [bottom(3,3:4) top(3,4:-1:3)]' [bottom(3,[4 1]) top(3,[1 4])]'];

            h = patch(Mx, My, Mz, colr, 'EdgeColor', 'none','FaceAlpha',0.4);
            

            %  shading interp
            h.FaceLighting = 'gouraud';
            h.BackFaceLighting = 'unlit';

        end
        
        
        function hs = calculateImage(hs,handidx)
            
            if nargin < 2
                handidx = find(hs.hands);
            end
            
            for i=1:size(handidx,2)
                
                j = handidx(i);
                Hand = hs.Hands{j};
                
                % calculate camPos relative to hand
                idxCams = sum(hs.hood.ptsCamSource,2) > 0;
                camPosesThisHand = hs.hood.camPos(:,idxCams);
                camPosesThisHandLocal = Hand.F' * camPosesThisHand;
                camPosThisHandLocal = mean(camPosesThisHandLocal,2);
                hs.imageCamPos(:,j) = camPosThisHandLocal;
                
                % If there are no local points, set image to zero and continue.
                if hs.hood.num() < 2 % less than two points
                    hs.images{j} = uint8(zeros(60,60,12));
                    continue;
                end
                
                % transform pts into rotated local Frenet frame
                croppedHoodPts = Hand.F'*(hs.hood.pts - repmat(Hand.sample,1,hs.hood.num()));
                croppedHoodNormals = Hand.F'*hs.hood.normals;
                
                [idxPts, croppedHoodPts] = Hand.getPts4UnitImage(croppedHoodPts,hs.imageparams);
                croppedHoodNormals = croppedHoodNormals(:,idxPts);
                
                % Calculate image for each of the following three view points.
                order = [1 2 3];
                [imgRGB12, imgDepth12] = Hand.getPtsImages(hs.imageparams,croppedHoodPts(order,:),croppedHoodNormals);
                order = [3 2 1];
                [imgRGB23, imgDepth23] = Hand.getPtsImages(hs.imageparams,croppedHoodPts(order,:),croppedHoodNormals);
                order = [3 1 2];
                [imgRGB31, imgDepth31] = Hand.getPtsImages(hs.imageparams,croppedHoodPts(order,:),croppedHoodNormals);
                
                % Concatenate all images into a single 15D image
                hs.images{j} = cat(3, ...
                    cat(3,imgRGB12,imgDepth12), ...
                    cat(3,imgRGB23,imgDepth23), ...
                    cat(3,imgRGB31,imgDepth31));
            end
        end
        



    end
    
end

