load('mat/H2SO4_H2O_log.mat')
load('mat/H2SO4_H2O_CRP94.mat')
ions = cellstr(ions);

% KH2O = KH2O .^ 2;
% KH2SO4 = KH2SO4 .^ 2;

meshclr = 0.8*[1 1 1];

figure(1); clf

subplot(2,3,1); hold on

    mesh(p_mOH,dHSO4,zeros(size(p_mOH)), 'facecolor','none', ...
        'edgecolor',meshclr)

    surf(p_mOH,dHSO4,KH2SO4, 'edgealpha',0.5)
    
    contour3(p_mOH,dHSO4,KH2SO4,[0 0], 'color','r', 'linewidth',1)
    
    sp.s1 = gca;

    plot3(solv(1)*[1 1],solv(2)*[1 1],get(gca,'zlim'),'r', ...
        'linewidth',3, 'marker','o')
    
    title('Keq H2SO4')
    
subplot(2,3,2); hold on

    mesh(p_mOH,dHSO4,zeros(size(p_mOH)), 'facecolor','none', ...
        'edgecolor',meshclr)

    
    surf(p_mOH,dHSO4,KH2O, 'edgealpha',0.5)
    
    contour3(p_mOH,dHSO4,KH2O,[0 0], 'color','r', 'linewidth',1)
    
    sp.s2 = gca;
    
    plot3(solv(1)*[1 1],solv(2)*[1 1],get(gca,'zlim'),'r', ...
        'linewidth',3, 'marker','o')
    
    title('Keq H2O')
    
subplot(2,3,3); hold on

    mesh(p_mOH,dHSO4,zeros(size(p_mOH)), 'facecolor','none', ...
        'edgecolor',meshclr)

    surf(p_mOH,dHSO4,Keq, 'edgealpha',0.5)
    
    plot(get(gca,'xlim'),solv(2)*[1 1],'r', 'linewidth',1)
    plot(solv(1)*[1 1],get(gca,'ylim'),'r', 'linewidth',1)
        
    sp.s3 = gca;
    
    plot3(solv(1)*[1 1],solv(2)*[1 1],get(gca,'zlim'),'r', ...
        'linewidth',3, 'marker','o')
    
%     zlim([0 1e-5])
    
    
for S = 1:3
    
    spl = ['s' num2str(S)];
    
    sp.(spl).XLabel.String = 'pOH';
    sp.(spl).YLabel.String = 'dHSO4';
    
    sp.(spl).View = [120 50];
    
end %for S

subplot(2,3,4); hold on

    plot(tots,Ksolved(:,1), 'marker','o')

subplot(2,3,5); hold on
    
    plot(tots,1-Ksolved(:,2), 'marker','o')
    
subplot(2,3,6); hold on

    plot(tots,mols(:,3))
    plot(tots,mols(:,4))
    plot(tots,mols(:,5))
    plot(tots,mols(:,6))
    
    legend(ions(3:end), 'location','nw')
