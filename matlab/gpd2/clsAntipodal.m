classdef clsAntipodal
    
    properties
        
        % input
        pts;
        normals;
        
%         antiPts; % 1xn binary indicating whether each pt has at least one an antipodal pair
%         antiCenter; % 3x1 indicating center of pair
%         antiAxis; % 3x1 normal vector pointing along line between pts in pair.
        
%         halfGrasp;
%         fullGrasp;
    end
    
    methods
        
        function a = clsAntipodal(pts,normals)
            a.pts = pts;
            a.normals = normals;
        end
        
        
        % Evaluate where this grasp is antipodal.
        % input: frictioncoeff -> angle of friction cone to be used in
        %                       evaluating the presence of a grasp (degrees).
        % output: fullGrasp -> this hand IS a grasp (binary). If fullGrasp
        %                       is 1, then halfGrasp must also be 1.
        function [fullGrasp, halfGrasp] = evalAntipodal(a,extremalthres)
            
            halfGrasp = 0;
            fullGrasp = 0;
            
%             frictioncoeff = 5; % degrees
            extremalthres = 0.04;
            frictioncoeff = 15; % degrees
            numViableThres = 10;
            
            pts = a.pts;
            normals = a.normals;

            % Calculate pts with surface normal w/in <frictioncoeff> of
            % closing direction.
            leftCloseDirection = [0 -1 0] * normals > cos(frictioncoeff*pi/180);
            rightCloseDirection = [0 1 0] * normals > cos(frictioncoeff*pi/180);

            % Calculate extremal pts
            leftExtremal = pts(2,:) < (min(pts(2,:)) + extremalthres);
            rightExtremal = pts(2,:) > (max(pts(2,:)) - extremalthres);
            
            % Viable points are those that are extremal and have a surface
            % normal in the friction cone of the closing direction.
            leftPtsViable = pts(:,leftCloseDirection & leftExtremal);
            rightPtsViable = pts(:,rightCloseDirection & rightExtremal);

            if (size(leftPtsViable,2) > 0) || (size(rightPtsViable,2) > 0)
                halfGrasp = 1;
            end

            if (size(leftPtsViable,2) > 0) && (size(rightPtsViable,2) > 0) % if we have viable points on both sides

                topViableY = min(max(leftPtsViable(1,:)),max(rightPtsViable(1,:)));
                bottomViableY = max(min(leftPtsViable(1,:)),min(rightPtsViable(1,:)));

                topViableZ = min(max(leftPtsViable(3,:)),max(rightPtsViable(3,:)));
                bottomViableZ = max(min(leftPtsViable(3,:)),min(rightPtsViable(3,:)));

                numViableLeft = sum((leftPtsViable(1,:) >= bottomViableY) & (leftPtsViable(1,:) <= topViableY) & (leftPtsViable(3,:) >= bottomViableZ) & (leftPtsViable(3,:) <= topViableZ));
                numViableRight = sum((rightPtsViable(1,:) >= bottomViableY) & (rightPtsViable(1,:) <= topViableY) & (rightPtsViable(3,:) >= bottomViableZ) & (rightPtsViable(3,:) <= topViableZ));
                
                if (numViableLeft >= numViableThres) && (numViableRight >= numViableThres)
                    fullGrasp = 1;
                end

            end
            
        end
        
        
        function plotPtsNormals(a)
            
            plot3(a.pts(1,:),a.pts(2,:),a.pts(3,:),'bx');
%             plot(a.pts(1,:),a.pts(2,:),'bx');
            hold on;
            d = 0.02;
            plot3([a.pts(1,:); a.pts(1,:) + d*a.normals(1,:)], [a.pts(2,:); a.pts(2,:) + d*a.normals(2,:)], [a.pts(3,:); a.pts(3,:) + d*a.normals(3,:)]);
%             plot([a.pts(1,:); a.pts(1,:) + d*a.normals(1,:)], [a.pts(2,:); a.pts(2,:) + d*a.normals(2,:)]);
            axis equal;
            
        end
        
        function plot(a)
            ii = find(a.antiPts);
            antiPts = a.pts(:,ii);
            antiCenter = a.antiCenter(:,ii);
            antiAxis = a.antiAxis(:,ii);
            
            plot3(a.pts(1,:),a.pts(2,:),a.pts(3,:),'bx');
            hold on;
            plot3(antiCenter(1,:),antiCenter(2,:),antiCenter(3,:),'mx');
            r = 0.01;
            plot3([antiCenter(1,:) - r*antiAxis(1,:);antiCenter(1,:) + r*antiAxis(1,:)], ...
                [antiCenter(2,:) - r*antiAxis(2,:);antiCenter(2,:) + r*antiAxis(2,:)], ...
                [antiCenter(3,:) - r*antiAxis(3,:);antiCenter(3,:) + r*antiAxis(3,:)]);
            axis equal;
        end
        
    end
    
end

