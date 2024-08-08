#%% Boxplots of behviors across days
output_folder = 'Figures/'+model+'/Stereotypies_development/'
num_of_mice = 0
for t in trials:
    num_of_mice = np.max([num_of_mice,len(TM[t].keys())])
print(num_of_mice)
fraction_of_frames = np.empty((num_of_behaviors,num_of_trials,num_of_mice))
fraction_of_frames[:,:,:] = np.nan
clustered_fraction_of_frames = np.empty((2,num_of_trials,num_of_mice))
clustered_fraction_of_frames[:,:,:] = np.nan
t_idx = 0
for t in trials:
    m_idx = 0
    for m in TM[t]:
        print(t,m)
        predictions = TM[t][m]['merged']
        for b in range(len(behaviors)):
            fraction_of_frames[b, t_idx, m_idx] = np.count_nonzero(predictions == b) / predictions.size
        clustered_predictions = np.copy(predictions)
        clustered_predictions = np.where(clustered_predictions<=3,VDB,nVDB)
        clustered_fraction_of_frames[VDB, t_idx, m_idx] = np.count_nonzero(clustered_predictions == VDB) / predictions.size
        clustered_fraction_of_frames[nVDB, t_idx, m_idx] = np.count_nonzero(clustered_predictions == nVDB) / predictions.size
        m_idx+=1
    t_idx+=1

for b in range(num_of_behaviors):
    plt.figure()
    sns.boxplot(data = fraction_of_frames[b,:,:].T,orient='v',color= colors[b])
    plt.yticks(np.arange(0,1,0.1),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.ylabel('Fraction of frames')
    plt.xticks(np.arange(8),trials)
    plt.title('Fraction of frames, behavior: '+behaviors[b])
    plt.xlabel('Day')
    plt.savefig(output_folder+'fraction_of_frames_'+behaviors[b]+'.png',dpi=300)
    plt.close()


plt.figure()
sns.boxplot(data = clustered_fraction_of_frames[VDB,:,:].T,orient='v',color= '#730AFF',boxprops=dict(alpha=.5))
plt.yticks(np.arange(0,1,0.1),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('Fraction of frames')
plt.xticks(np.arange(8),trials)
plt.title('Fraction of frames, behaviors:Grooming, body licking,floor licking, wall licking')
plt.xlabel('Day')
plt.savefig(output_folder+'fraction_of_frames_VDB.png',dpi=300)
plt.close()

plt.figure()
sns.boxplot(data = clustered_fraction_of_frames[nVDB,:,:].T,orient='v',color= 'gray',boxprops=dict(alpha=.3))
plt.yticks(np.arange(0,1,0.1),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('Fraction of frames')
plt.xticks(np.arange(8),trials)
plt.title('Fraction of frames, behaviors:Rearing, other, BTC, jump')
plt.xlabel('Day')
plt.savefig(output_folder+'fraction_of_frames_nVDB.png',dpi=300)
plt.close()
#%% Bouts analysis for the different days
output_folder = 'Figures/'+model+'/Stereotypies_development/Bouts/'
t_idx = 0
num_of_mice = []
max_num_of_mice = 0
mice_list = set()
for t in trials:
    mice_list.update(list(TM[t].keys()))
    print(mice_list)
    max_num_of_mice = np.max([max_num_of_mice,len(TM[t].keys())])
    num_of_mice.append(len(TM[t].keys()))


IBI_mean =  np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
IBI_mean[:,:,:]=np.nan
length_mean = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
length_mean[:,:,:]=np.nan
number = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
number[:,:,:] = np.nan
IBI_stderr = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
IBI_stderr[:,:,:] = np.nan
length_stderr = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
length_stderr[:,:,:]=np.nan

for t in trials:
    m_idx = 0
    for m in mice_list:
        print(t,m)
        if m in TM[t]:
            predictions = TM[t][m]['merged']
            for b in range(num_of_behaviors):
                bouts_data = hf.segment_bouts(predictions,b,10)
                IBI_mean[m_idx,t_idx,b] = np.mean(bouts_data['IBI'])
                IBI_stderr[m_idx, t_idx, b] = np.sqrt(np.var(bouts_data['IBI'])/len(bouts_data['IBI']))
                length_mean[m_idx, t_idx, b] = np.mean(bouts_data['length'])
                length_stderr[m_idx, t_idx, b] = np.sqrt(np.var(bouts_data['length']) / len(bouts_data['length']))
                number[m_idx,t_idx,b] = bouts_data['number']
        m_idx += 1
    t_idx += 1
fig_idx = 1
for b in range(num_of_behaviors):
    for m_idx in range(max_num_of_mice):
        plt.figure(fig_idx)
        plt.plot(np.arange(num_of_trials),number[m_idx,:,b],marker='o',ls='--',markerfacecolor='white',lw=1,c=colors[b])
        plt.title('Number of bouts , behavior : '+behaviors[b])
        plt.xlabel('Day')
        plt.xticks(np.arange(num_of_trials),trials)
        plt.ylabel('# of bouts')
        plt.figure(fig_idx+1)
        plt.plot(np.arange(num_of_trials), length_mean[m_idx, :, b], marker='o', ls='--', markerfacecolor='white', lw=1, c=colors[b])
        plt.errorbar(np.arange(num_of_trials), y=length_mean[m_idx,:,b], yerr=length_stderr[m_idx,:,b], c=colors[b], capsize=2, capthick=1, ls='')
        plt.title('Bout duration , behavior : '+behaviors[b])
        plt.xlabel('Day')
        plt.xticks(np.arange(num_of_trials),trials)
        plt.ylabel('Bout duration (# of frames)')
        plt.figure(fig_idx+2)
        plt.plot(np.arange(num_of_trials), IBI_mean[m_idx, :, b], marker='o', ls='--', markerfacecolor='white', lw=1, c=colors[b])
        plt.errorbar(np.arange(num_of_trials), y=IBI_mean[m_idx, :, b], yerr=IBI_stderr[m_idx,:,b], c=colors[b], capsize=2, capthick=1, ls='')
        plt.title('Inter bout interval , behavior : '+behaviors[b])
        plt.xlabel('Day')
        plt.xticks(np.arange(num_of_trials),trials)
        plt.ylabel('Inter-bout-interval (# of frames)')

    plt.figure(fig_idx)
    plt.savefig(output_folder+'number_of_bouts_'+behaviors[b]+'.png',dpi=300)
    plt.close(fig_idx)
    plt.figure(fig_idx+1)
    plt.savefig(output_folder+'bout_duration_'+behaviors[b]+'.png',dpi=300)
    plt.close(fig_idx+1)
    plt.figure(fig_idx+2)
    plt.savefig(output_folder+'IBI_'+behaviors[b]+'.png',dpi=300)
    plt.close(fig_idx+2)
    fig_idx+=3

#%% Bouts analysis for the different days - normelized # of bouts, and CDF of bout_duration
output_folder = 'Figures/'+model+'/Stereotypies_development/Bouts/'
t_idx = 0
num_of_mice = []
max_num_of_mice = 0
mice_list = set()
for t in trials:
    mice_list.update(list(TM[t].keys()))
    print(mice_list)
    max_num_of_mice = np.max([max_num_of_mice,len(TM[t].keys())])
    num_of_mice.append(len(TM[t].keys()))


IBI_mean =  np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
IBI_mean[:,:,:]=np.nan
length_mean = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
length_mean[:,:,:]=np.nan
number = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
number[:,:,:] = np.nan
IBI_stderr = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
IBI_stderr[:,:,:] = np.nan
length_stderr = np.empty((max_num_of_mice,num_of_trials,num_of_behaviors))
length_stderr[:,:,:]=np.nan
length = {}
for t in trials:
    m_idx = 0
    length[t] = {}
    for m in mice_list:
        print(t,m)
        if m in TM[t]:
            predictions = TM[t][m]['merged']

            for b in range(num_of_behaviors):

                bouts_data = hf.segment_bouts(predictions,b,10)
                # IBI_mean[m_idx,t_idx,b] = np.mean(bouts_data['IBI'])
                # IBI_stderr[m_idx, t_idx, b] = np.sqrt(np.var(bouts_data['IBI'])/len(bouts_data['IBI']))
                # length_mean[m_idx, t_idx, b] = np.mean(bouts_data['length'])
                # length_stderr[m_idx, t_idx, b] = np.sqrt(np.var(bouts_data['length']) / len(bouts_data['length']))
                number[m_idx,t_idx,b] = bouts_data['number']/np.count_nonzero(predictions==b)


        m_idx += 1
    t_idx += 1
fig_idx = 1
for b in range(num_of_behaviors):
    for m_idx in range(max_num_of_mice):
        plt.figure(fig_idx)
        plt.plot(np.arange(num_of_trials),number[m_idx,:,b],marker='o',ls='--',markerfacecolor='white',lw=1,c=colors[b])
        plt.title('Number of bouts , behavior : '+behaviors[b])
        plt.xlabel('Day')
        plt.xticks(np.arange(num_of_trials),trials)
        plt.ylabel('# of bouts')
        plt.figure(fig_idx+1)
        plt.plot(np.arange(num_of_trials), length_mean[m_idx, :, b], marker='o', ls='--', markerfacecolor='white', lw=1, c=colors[b])
        plt.errorbar(np.arange(num_of_trials), y=length_mean[m_idx,:,b], yerr=length_stderr[m_idx,:,b], c=colors[b], capsize=2, capthick=1, ls='')
        plt.title('Bout duration , behavior : '+behaviors[b])
        plt.xlabel('Day')
        plt.xticks(np.arange(num_of_trials),trials)
        plt.ylabel('Bout duration (# of frames)')
        plt.figure(fig_idx+2)
        plt.plot(np.arange(num_of_trials), IBI_mean[m_idx, :, b], marker='o', ls='--', markerfacecolor='white', lw=1, c=colors[b])
        plt.errorbar(np.arange(num_of_trials), y=IBI_mean[m_idx, :, b], yerr=IBI_stderr[m_idx,:,b], c=colors[b], capsize=2, capthick=1, ls='')
        plt.title('Inter bout interval , behavior : '+behaviors[b])
        plt.xlabel('Day')
        plt.xticks(np.arange(num_of_trials),trials)
        plt.ylabel('Inter-bout-interval (# of frames)')

    plt.figure(fig_idx)
    plt.savefig(output_folder+'number_of_bouts_'+behaviors[b]+'.png',dpi=300)
    plt.close(fig_idx)
    plt.figure(fig_idx+1)
    plt.savefig(output_folder+'bout_duration_'+behaviors[b]+'.png',dpi=300)
    plt.close(fig_idx+1)
    plt.figure(fig_idx+2)
    plt.savefig(output_folder+'IBI_'+behaviors[b]+'.png',dpi=300)
    plt.close(fig_idx+2)
    fig_idx+=3




#%% Dynamics of boutiness
output_folder = 'Figures/'+model+'/Stereotypies_development/Bouts/'
T = 25
num_of_mice = []
for t in trials:
    num_of_mice.append(len(TM[t].keys()))
max_num_of_mice = np.max(num_of_mice)
num_of_bouts = np.zeros((max_num_of_mice,num_of_trials,num_of_behaviors,T))
num_of_bouts[:,:,:,:]=np.nan
noNan_mice =  np.zeros((num_of_trials,num_of_behaviors,T))+max_num_of_mice
tot_boutiness = np.zeros((max_num_of_mice,num_of_trials,num_of_behaviors))
tot_boutiness[:,:,:]=np.nan
for tr_idx in range(num_of_trials):
    for b in range(num_of_behaviors):
        m_idx = 0
        tr = trials[tr_idx]
        for m in TM[tr]:
            predictions = TM[tr][m]['merged']
            print(behaviors[b],m)
            for t in np.arange(0,T*minute,minute):
                bouts = hf.segment_bouts(predictions[:t], b, 0)
                if  np.count_nonzero(predictions[:t] == b)>0:
                    num_of_bouts[m_idx, tr_idx, b,int(t/minute)] = 1 - (bouts['number'] / np.count_nonzero(predictions[:t] == b))
                else:
                    num_of_bouts[m_idx, tr_idx, b,int(t/minute)] = np.nan
                    noNan_mice[tr_idx,b,int(t/minute)]-=1
            if np.count_nonzero(predictions==b)>0:
                bouts = hf.segment_bouts(predictions, b, 0)
                tot_boutiness[m_idx,tr_idx,b]=  1 - (bouts['number'] / np.count_nonzero(predictions == b))
            m_idx+=1
print(tot_boutiness)
colors_prime = ['gray','gray','gray','blue','turquoise','green','orange','red']
alpha_prime = [0.33,0.66,0.5,0.5,0.5,0.5,0.5,0.5]
for b in range(num_of_behaviors):
    plt.figure()
    for t_idx in range(num_of_trials):
        mean = np.nanmean(num_of_bouts[:,t_idx,b,:], axis=0)
        stderr = np.sqrt(np.nanvar(num_of_bouts[:,t_idx,b,:],axis=0)/noNan_mice[t_idx,b])
        plt.plot(mean, alpha=alpha_prime[t_idx], color=colors_prime[t_idx], marker='o', label=trials[t_idx], lw=1)
        plt.errorbar(np.arange(T), y=mean, yerr=stderr, color=colors_prime[t_idx], capsize=2, capthick=.5, alpha=alpha_prime[t_idx], ls='', lw=1)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    plt.ylim([0.5, 1])
    plt.ylabel('Boutiness (1- $\\frac{|B|}{|F|}$)')
    plt.title('Change Along the development of stereotypies\n Behavior: '+behaviors[b])
    plt.legend()
    plt.xticks(np.arange(T), np.arange(T))
    plt.xlabel('Time(m)')
    plt.savefig(output_folder + '/dynamics_boutiness_'+behaviors[b]+'.png', dpi=300)
    plt.close()

plt.figure()
for b in range(num_of_behaviors):
    mean = np.nanmean(tot_boutiness[:, :, b], axis=0)
    stderr = np.sqrt(np.nanvar(tot_boutiness[:, :, b], axis=0) / num_of_mice)
    plt.plot(mean, alpha=.5, color=colors[b], marker='o', label=behaviors[b], lw=1)
    plt.errorbar(np.arange(num_of_trials), y=mean, yerr=stderr, color=colors[b], capsize=2, capthick=.5, alpha=0.5, ls='', lw=1)

axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0.5, 1, 1.2], [0.5, 1, 1.2])
plt.ylim([0.5, 1.2])
plt.ylabel('Boutiness (1- $\\frac{|B|}{|F|}$)')
plt.title('Change Along the development of stereotypies')
plt.legend()
plt.xticks(np.arange(num_of_trials), trials)
plt.xlabel('Day')
plt.savefig(output_folder + '/dynamics_totBoutiness_'+behaviors[b]+'.png', dpi=300)
plt.close()
#%%V graphs on bins
output_folder = 'Figures/'+model+'/DREADDS_thin/'+cohort+'/Summary/'
num_of_bins = 3
fraction_of_frames = np.zeros((num_of_mice , num_of_behaviors , num_of_days,num_of_bins))
clustered_fraction_of_frames = np.zeros((num_of_mice , 2 , num_of_days,num_of_bins))
m_idx = 0
for m in sandwich_days:
    pre = np.copy(sandwich_days[m]['pre']['merged'])
    edges_pre = np.linspace(0,pre.size,num_of_bins+1,dtype=int)
    print(edges_pre)
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    edges_CNO = np.linspace(0, CNO.size, num_of_bins+1, dtype=int)
    print(edges_CNO)
    post = np.copy(sandwich_days[m]['post']['merged'])
    edges_post = np.linspace(0, post.size, num_of_bins+1, dtype=int)
    print(edges_post)
    for b in range(num_of_behaviors):
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end-bin_start
            fraction_of_frames[m_idx , b , PRE,edge] = np.count_nonzero(pre[bin_start:bin_end]==b)/bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge + 1]
            bin_size = bin_end - bin_start
            fraction_of_frames[m_idx, b, DURING,edge] = np.count_nonzero(CNO[bin_start:bin_end] == b) / bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge + 1]
            bin_size = bin_end - bin_start
            fraction_of_frames[m_idx, b, POST,edge] = np.count_nonzero(post[bin_start:bin_end] == b) / bin_size
    clustered_pre = np.where(pre<=3,VDB,nVDB)
    clustered_CNO = np.where(CNO<=3,VDB,nVDB)
    clustered_post = np.where(post<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end-bin_start
            clustered_fraction_of_frames[m_idx,category,PRE,edge] = np.count_nonzero(clustered_pre[bin_start:bin_end]==category)/bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge+1]
            bin_size = bin_end-bin_start
            clustered_fraction_of_frames[m_idx, category, DURING,edge] = np.count_nonzero(clustered_CNO[bin_start:bin_end] == category) / bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge+1]
            bin_size = bin_end-bin_start
            clustered_fraction_of_frames[m_idx, category, POST,edge] = np.count_nonzero(clustered_post[bin_start:bin_end] == category) / bin_size
    m_idx+=1

for b in range(num_of_behaviors):
    fig,ax = plt.subplots(ncols=3,sharey='all')
    for bin in range(num_of_bins):
        plt.sca(ax[bin])
        mean = np.mean(fraction_of_frames[:,b,:,bin],axis=0)
        stderr = np.sqrt(np.var(fraction_of_frames[:, b, :,bin], axis=0)/num_of_mice)
        plt.plot(fraction_of_frames[:,b,:,bin].T, alpha=0.3, color='gray', marker='o')
        plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
        plt.plot(mean.T, color=colors[b])
        plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=1)
        plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
        plt.yticks([0,0.5,1],[0,0.5,1])
        plt.ylim([0,1])
        axes = plt.gca()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.title('Bin #'+str(bin))
    ax[0].set_ylabel('Fraction of frames')
    plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: '+behaviors[b])
    plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
    plt.savefig(output_folder + '/bins_V_graph_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

fig,ax = plt.subplots(ncols=3,sharey='all')
for bin in range(num_of_bins):
    plt.sca(ax[bin])
    mean = np.mean(clustered_fraction_of_frames[:,VDB,:,bin],axis=0)
    stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,VDB,:,bin], axis=0)/num_of_mice)
    plt.plot(clustered_fraction_of_frames[:,VDB,:,bin].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean, color='#730AFF')
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])

    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Bin #' + str(bin))
ax[0].set_ylabel('Fraction of frames')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Grooming, body licking , floor licking , wall licking')
plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
plt.savefig(output_folder + '/bins_V_graph_VDB.png', dpi=300)
plt.close()

fig,ax = plt.subplots(ncols=3,sharey='all')
for bin in range(num_of_bins):
    plt.sca(ax[bin])
    mean = np.mean(clustered_fraction_of_frames[:,nVDB,:,bin],axis=0)
    stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,nVDB,:,bin], axis=0)/num_of_mice)
    plt.plot(clustered_fraction_of_frames[:,nVDB,:,bin].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean, color='#730AFF')
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Bin #' + str(bin))
ax[0].set_ylabel('Fraction of frames')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Rearing, BTC, other, jump')
plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
plt.savefig(output_folder + '/bins_V_graph_nVDB.png', dpi=300)
plt.close()
#%% Heatmaps and means of behavior related transients (BRTs) - only mice sampled at above 30Hz
output_folder = 'Figures/'+model+'/Photometry/'
np_sig = {'left':{'short':[],'long':[]},'right':{'short':[],'long':[]}}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds

for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={'long':[],'short':[]}
            bout_length[pathway][b][tr]={'long':[],'short':[]}
for c in CMT:
    for m in CMT[c]:
        if m not in mice_list:continue
        if c in ['drd1_hm4di', 'drd1_hm3dq']:
            pathway = 'drd1'
        elif c in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
            pathway = 'a2a'
        elif c == 'controls':
            if m.startswith('cA'):
                pathway = 'a2a'
            else:
                pathway = 'drd1'
        else:
            break
        for tr in CMT[c][m]:
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']
                    right = CMT[c][m][tr]['photom']['right']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']

                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                    for length in ['short', 'long']:
                        for bout_idx in range(len(np_bouts[length])):
                            # print(len(np_bouts[length]) , bout_idx)
                            if 'saline' in tr: tr = 'saline'
                            if side== 'left':
                                if np_bouts[length][bout_idx][0] >= 5*seconds and left.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_left = left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds] - np.mean(
                                        left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(sig_left)

                            elif side == 'right':
                                if np_bouts[length][bout_idx][0] >= 5*seconds and right.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_right = right[np_bouts[length][bout_idx][0] - 5 * seconds:np_bouts[length][bout_idx][0] + 10 * seconds] - np.mean(
                                        right[np_bouts[length][bout_idx][0] - 5 * seconds:np_bouts[length][bout_idx][0] - 2 * seconds])
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(sig_right)
                            else:
                                if np_bouts[length][bout_idx][0] >= 5*seconds and left.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_left = left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds] - np.mean(
                                        left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])


                                    sig_right = right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds] - np.mean(
                                        right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])

                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(sig_left)
                                    photometry[pathway][b][tr][length].append(sig_right)

cohorts = ['drd1','a2a']
behaviors_prime = [0,1,2,3,4,6]
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=2, ncols=7, sharex=True, figsize=(20,9))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            long = np.array(photometry[c][b][tr]['long'])
            short = np.array(photometry[c][b][tr]['short'])
            if long.shape[0] ==0 and short.shape[0]==0 :
                tr_idx+=1
                continue
            n_long = long.shape[0]
            if long.shape[0]>0:
                mean = np.nanmean(long,axis=0)
                stderr = np.sqrt(np.nanvar(long, axis=0)/long.shape[0])
                ax[0,tr_idx].plot(mean, label='Long (n='+str(n_long)+')',c=c1, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c1, alpha=low_alpha)

            n_short = short.shape[0]
            if short.shape[0]>0:
                mean = np.nanmean(short,axis=0)
                stderr = np.sqrt(np.nanvar(short, axis=0)/short.shape[0])
                ax[0, tr_idx].plot(mean, label='Short (n='+str(n_short)+')', c=c2, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c2, alpha=high_alpha)
            n_str = 'N$_{short}$='+str(n_short)+', N$_{long}$='+str(n_long)
            print(long.shape, short.shape)
            if short.shape[0]>0 and long.shape[0] == 0:
                all_events = np.copy(short)
            elif short.shape[0] == 0 and long.shape[0]>0:
                all_events = np.copy(long)
            else:
                all_events = np.vstack((short,long))
            if tr_idx==0:
                ax[0,tr_idx].set_ylabel('Mean $\\Delta$F/F (Z-score)',fontsize=12)
            ax[0,tr_idx].set_ylim([-.7, 2])
            ax[0,tr_idx].vlines(x=bout_onset, ymin=-.7,ymax=2, ls='dashed', colors='k')
            ax[0,tr_idx].legend()
            plt.sca(ax[0,tr_idx])
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds), '')
            bout_length[c][b][tr]['short'].extend(bout_length[c][b][tr]['long'])
            events_length = np.array(bout_length[c][b][tr]['short'])
            indices = np.argsort(events_length)
            # print(long.shape , short.shape , all_events.shape , indices.size , events_length.shape)
            ax[1, tr_idx] = sns.heatmap(all_events[indices,], vmin=lims[sig_type][cohort][0], vmax=lims[sig_type][cohort][1], ax=ax[1, tr_idx],cbar=False)
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[1,tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds),
                       ((np.arange(0, window_length + 1, 2 * seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])
            plt.title(tr)
            tr_idx+=1
        plt.subplots_adjust(left = 0.1 , bottom =0.05, right = .95, top =  0.9)
        plt.suptitle(c+';  $\\theta$$_{short}^{long}$ = 2s ; Quiet BL period = 3s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/'+ c + '/summary_above30HzOnly/'+behaviors[b]+'_heatmapsAndMeans.png',dpi=300)
        plt.close()

#%% Per mouse transients - only mice sampled at above 30 Hz
cohort = 'drd1'
if cohort == 'drd1':
    cohorts = ['drd1_hm4di','drd1_hm3dq','controls']
else:
    cohorts = ['a2a_hm4di','a2a_hm3dq','controls']
output_folder = 'Figures/'+model+'/Photometry/'+cohort+'/perMouse_above30HzOnly/'
all_mice_output_folder = 'Figures/'+model+'/Photometry/'+cohort+'/allMice_above30HzOnly/'
np_sig = {'left':{'short':[],'long':[]},'right':{'short':[],'long':[]}}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 1
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 10*seconds
window_right_wing = 5*seconds # relative to bout onset
window_left_wing = 5*seconds # relative to bout onset
bout_onset = 5*seconds
behaviors_prime = [0,1,2,3,4,6]
length = 'long'

for b in behaviors_prime:
    photometry = {}
    bout_length = {}
    for tr in ['saline1', 'saline2', 'saline3']:
        print(tr)
        for c in cohorts:
            if tr not in CTM[c]: continue
            for m in CTM[c][tr]:
                if m not in mice_list:continue
                if cohort=='drd1':
                    if  c =='controls' and m.startswith('cA'):continue # for drd1
                else:
                    if c == 'controls' and not m.startswith('cA'): continue  # for a2a
                if not 'photom' in CMT[c][m][tr].keys():continue
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom']:
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']
                    right = CMT[c][m][tr]['photom']['right']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']
                predictions = CMT[c][m][tr]['merged']
                if m not in photometry :
                    photometry[m] = []
                    bout_length[m] = []
                np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                for bout_idx in range(len(np_bouts[length])):
                    if side == 'left':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)

                    elif side == 'right':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and right.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_right = right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_right)
                    else:
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            sig_right = right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)
                            photometry[m].append(sig_right)
    num_of_mice = len(photometry.keys())

    for m in photometry:
        if len(photometry[m]) == 0:
            num_of_mice -= 1
        else:
            transients = np.copy(np.array(photometry[m]))
            transients = transients[~np.isnan(transients).all(axis=1)]
            if transients.shape[0] == 0:  # in  case all rows were nan moues should be omitted
                num_of_mice -= 1
    if num_of_mice > 0:
        hr = [3]
        for m_idx in range(num_of_mice):
            hr.append(1)
        fig , ax = plt.subplots(nrows = num_of_mice+1 , sharex = 'all' ,gridspec_kw={'height_ratios': hr}, figsize=(4,12))
        row_idx=1
        all_mice_transients = np.zeros(window_length)
        all_mice_lengths = []
        for m in photometry:
            events_length = np.array(bout_length[m])
            indices = np.argsort(events_length)
            transients = np.array(photometry[m])
            if transients.shape[0] == 0: continue
            transients = transients[indices,]
            all_mice_transients = np.vstack([all_mice_transients, transients])
            transients = transients[~np.isnan(transients).all(axis=1)]
            if transients.shape[0] == 0: # in  case all rows were nan moues should be ommited
                continue
            all_mice_lengths.extend(bout_length[m])

            plt.sca(ax[row_idx])
            ax[row_idx] = sns.heatmap(transients, vmin=-0.7, vmax=2, ax=ax[row_idx], cbar=False)
            plt.yticks(np.linspace(0,transients.shape[0],3,dtype=int),np.linspace(0,transients.shape[0],3,dtype=int))
            plt.ylabel(m)
            plt.sca(ax[0])
            mean = np.nanmean(transients,axis=0)
            stderr = np.sqrt(np.nanvar(transients,axis=0)/transients.shape[0])
            plt.plot(mean,label = m,alpha=high_alpha)
            plt.fill_between(x = np.arange(window_length),y1 = mean-stderr,y2=mean+stderr,alpha=low_alpha)
            plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
            plt.ylabel('$\Delta{F}/{F}(Z)$')
            row_idx+=1
        plt.sca(ax[0])

        plt.legend(loc = 'upper left' , ncol = int(np.ceil(num_of_mice/2)))
        plt.sca(ax[num_of_mice])
        plt.xticks(np.arange(0, window_length + 1,  seconds),
                   ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.xlabel('Time relative to bout onset')
        plt.suptitle('Saline - ' + behaviors[b])
        plt.subplots_adjust(left=0.2, bottom=0.05, right=.9, top=0.9)
        plt.savefig(output_folder+'saline/'+behaviors[b]+'.png')
        plt.close()

    all_mice_events_length = np.array(all_mice_lengths)
    sorted_indices = np.argsort(all_mice_events_length)
    all_mice_transients = all_mice_transients[1:,:]
    all_mice_transients = all_mice_transients[sorted_indices,]
    all_mice_transients =  all_mice_transients[~np.isnan( all_mice_transients).all(axis=1)]
    mean = np.nanmean(all_mice_transients,axis=0)
    stderr = np.sqrt(np.nanvar(all_mice_transients,axis=0)/all_mice_transients.shape[0])
    fig,ax = plt.subplots(nrows=2,sharex='all',figsize=(4,12))
    plt.sca(ax[1])
    ax[1] = sns.heatmap(all_mice_transients, vmin=-0.7, vmax=2, ax=ax[1], cbar=False)
    plt.yticks(np.linspace(0, all_mice_transients.shape[0], 3, dtype=int), np.linspace(0, all_mice_transients.shape[0], 3, dtype=int))
    plt.xticks(np.arange(0, window_length + 1, seconds),
               ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
    plt.xlabel('Time relative to bout onset')
    plt.ylabel('# of events')
    plt.sca(ax[0])
    plt.plot(mean, alpha=high_alpha)
    plt.fill_between(x=np.arange(window_length), y1=mean - stderr, y2=mean + stderr, alpha=low_alpha)
    plt.ylim([-0.5, 2])
    plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
    plt.ylabel('$\Delta{F}/{F}(Z)$')
    plt.suptitle(cohort + ' ; Saline Days ; '+behaviors[b] )
    plt.savefig(all_mice_output_folder + 'saline/' + behaviors[b] + '.png')
    plt.subplots_adjust(left=0.2, bottom=0.05, right=.9, top=0.9)
    plt.close()


for tr in ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5', 'splashTest']:
    print(tr)
    for b in behaviors_prime:
        photometry = {}
        bout_length = {}
        for c in cohorts:
            if tr not in CTM[c]: continue
            for m in CTM[c][tr]:
                if m not in mice_list: continue
                if cohort=='drd1':
                    if c =='controls' and m.startswith('cA') : continue # for drd1
                else:
                    if c == 'controls' and not m.startswith('cA'): continue  # for a2a
                if not 'photom' in CMT[c][m][tr].keys():continue
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom']:
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']
                    right = CMT[c][m][tr]['photom']['right']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']
                predictions = CMT[c][m][tr]['merged']
                photometry[m] = []
                bout_length[m] = []
                np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                for bout_idx in range(len(np_bouts[length])):
                    if side == 'left':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[
                                       np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)

                    elif side == 'right':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and right.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_right = right[
                                        np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_right)
                    else:
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[
                                       np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            sig_right = right[
                                        np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)
                            photometry[m].append(sig_right)

        num_of_mice = len(photometry.keys())
        num_of_events = 0
        for m in photometry:
            num_of_events+=len(photometry[m])
            if len(photometry[m]) == 0:
                num_of_mice -= 1
            else:
                transients = np.copy(np.array(photometry[m]))
                transients = transients[~np.isnan(transients).all(axis=1)]
                if transients.shape[0] == 0:  # in  case all rows were nan moues should be omitted
                    num_of_mice -= 1
        print(tr,behaviors[b],num_of_events)
        if num_of_mice==0: continue
        hr = [3]
        for m_idx in range(num_of_mice):
            hr.append(1)
        all_mice_transients = np.zeros(window_length)
        all_mice_lengths = []
        fig , ax = plt.subplots(nrows = num_of_mice+1,sharex = 'all',gridspec_kw={'height_ratios': hr}, figsize=(4,12))
        row_idx=1
        for m in photometry:
            events_length = np.array(bout_length[m])
            indices = np.argsort(events_length)
            transients = np.array(photometry[m])
            if transients.shape[0] == 0: continue
            transients = transients[indices,]
            all_mice_transients = np.vstack([all_mice_transients, transients])
            transients = transients[~np.isnan(transients).all(axis=1)]
            if transients.shape[0] == 0:# in  case all rows were nan moues should be ommited
                continue
            all_mice_lengths.extend(bout_length[m])
            plt.sca(ax[row_idx])
            ax[row_idx] = sns.heatmap(transients, vmin=-0.7, vmax=2, ax=ax[row_idx], cbar=False)
            plt.yticks(np.linspace(0,transients.shape[0],3,dtype=int),np.linspace(0,transients.shape[0],3,dtype=int),rotation='horizontal')
            plt.ylabel(m)
            plt.sca(ax[0])
            mean = np.nanmean(transients,axis=0)
            stderr = np.sqrt(np.nanvar(transients,axis=0)/transients.shape[0])
            plt.plot(mean,label = m,alpha=high_alpha)
            plt.fill_between(x = np.arange(window_length),y1 = mean-stderr,y2=mean+stderr,alpha=low_alpha)
            plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
            plt.ylabel('$\Delta{F}/{F}(Z)$')
            row_idx+=1
        plt.sca(ax[0])
        plt.legend(loc = 'upper left' , ncol = int(np.ceil(num_of_mice/2)))
        plt.sca(ax[num_of_mice])
        plt.xlabel('Time relative to bout onset')
        plt.xticks(np.arange(0, window_length + 1,  seconds),
                   ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.suptitle(tr + '-' + behaviors[b])
        plt.subplots_adjust(left=0.2, bottom=0.05, right=.9, top=0.9)
        plt.savefig(output_folder+tr+'/'+behaviors[b]+'.png')
        plt.close()
        all_mice_events_length = np.array(all_mice_lengths)
        sorted_indices = np.argsort(all_mice_events_length)
        all_mice_transients = all_mice_transients[1:, :]
        all_mice_transients = all_mice_transients[sorted_indices,]
        all_mice_transients = all_mice_transients[~np.isnan(all_mice_transients).all(axis=1)]
        mean = np.nanmean(all_mice_transients, axis=0)
        stderr = np.sqrt(np.nanvar(all_mice_transients, axis=0) / all_mice_transients.shape[0])
        fig, ax = plt.subplots(nrows=2, sharex='all',figsize=(4,12))
        plt.sca(ax[1])
        ax[1] = sns.heatmap(all_mice_transients, vmin=-0.7, vmax=2, ax=ax[1], cbar=False)
        plt.yticks(np.linspace(0, all_mice_transients.shape[0], 3, dtype=int), np.linspace(0, all_mice_transients.shape[0], 3, dtype=int))
        plt.xticks(np.arange(0, window_length + 1, seconds),
                   ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.xlabel('Time relative to bout onset')
        plt.ylabel('# of events')
        plt.sca(ax[0])
        plt.plot(mean, alpha=high_alpha)
        plt.fill_between(x=np.arange(window_length), y1=mean - stderr, y2=mean + stderr, alpha=low_alpha)
        plt.ylim([-0.5,2])
        plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
        plt.ylabel('$\Delta{F}/{F}(Z)$')
        plt.suptitle(cohort + ' ; '+ tr+' ; ' + behaviors[b])
        plt.subplots_adjust(left=0.2, bottom=0.05, right=.9, top=0.9)
        plt.savefig(all_mice_output_folder + tr+'/' + behaviors[b] + '.png')
        plt.close()


#%% Histograms of the time to peak from behavioral transition
window_to_transient = 5*FPS
prom = 2
bout_theta = 1
output_folder = 'Figures/'+model+'/Photometry/'
time_to_peak = {}
time_to_trough = {}
for pathway in ['drd1','a2a']:
    time_to_peak[pathway]={}
    time_to_trough[pathway]={}
    for tr in ['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5','splashTest']:
        time_to_peak[pathway][tr]={}
        time_to_trough[pathway][tr]={}
        for b_cur in range(num_of_behaviors):
            time_to_peak[pathway][tr][b_cur]={}
            time_to_trough[pathway][tr][b_cur] = {}
            for b_next in range(num_of_behaviors):
                time_to_peak[pathway][tr][b_cur][b_next] = []
                time_to_trough[pathway][tr][b_cur][b_next] = []
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={'long':[],'short':[]}
            bout_length[pathway][b][tr]={'long':[],'short':[]}
for c in CMT:
    for m in CMT[c]:
        if c in ['drd1_hm4di', 'drd1_hm3dq']:
            pathway = 'drd1'
        elif c in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
            pathway = 'a2a'
        elif c == 'controls':
            if m.startswith('cA'):
                pathway = 'a2a'
            else:
                pathway = 'drd1'
        else:
            break
        for tr in CMT[c][m]:
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                predictions = CMT[c][m][tr]['merged']
                pi = hf.segement_bouts_transition(predictions, bout_theta * FPS)
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']
                    right = CMT[c][m][tr]['photom']['right']
                    if 'saline' in tr: tr = 'saline'
                    for sig in [left , right]:
                        peaks = signal.find_peaks(sig , prominence=prom)[0]
                        troughs = signal.find_peaks(-sig, prominence=prom)[0]
                        for b_cur in range(num_of_behaviors):
                            for b_next in range(num_of_behaviors):
                                for idx in pi[b_cur][b_next]:
                                    if peaks.size > 0:
                                        closest_peak = np.min(np.abs(idx-peaks))
                                        real_diff = idx - peaks[np.argmin(np.abs(idx-peaks))]
                                        if closest_peak < window_to_transient:
                                            time_to_peak[pathway][tr][b_cur][b_next].append(real_diff)
                                    if troughs.size>0:
                                        closest_trough = np.min(np.abs(idx - troughs))
                                        real_diff = idx - troughs[np.argmin(np.abs(idx - troughs))]
                                        if closest_trough < window_to_transient:
                                            time_to_trough[pathway][tr][b_cur][b_next].append(real_diff)


# exit(0)
trials_prime = ['saline', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5', 'splashTest']
cohorts = ['drd1', 'a2a']
for c in cohorts:
    for b_cur in range(num_of_behaviors):
        for b_next in range(num_of_behaviors):
            if b_cur == b_next : continue
            fig, ax = plt.subplots(nrows=2, ncols=len(trials_prime),sharex='all',figsize=(20,6))
            tr_idx=0
            for tr in trials_prime:
                plt.sca(ax[0,tr_idx])

                sns.histplot(time_to_peak[c][tr][b_cur][b_next], alpha=.5,color = 'red',stat = 'probability',bins = np.arange(-window_to_transient,window_to_transient,5))
                if tr_idx==0:
                    plt.ylabel('Peaks')
                else:
                    plt.ylabel('')
                plt.title(tr)
                plt.sca((ax[1,tr_idx]))
                sns.histplot(time_to_trough[c][tr][b_cur][b_next], alpha=.5 , color = 'green',stat = 'probability',bins = np.arange(-window_to_transient,window_to_transient,5))
                if tr_idx==0:
                    plt.ylabel('Troughs')
                else:
                    plt.ylabel('')
                if tr_idx==2:
                    plt.xlabel('Time relative to transition(s)')
                plt.xticks(np.arange(-window_to_transient,window_to_transient,FPS),(np.arange(-window_to_transient,window_to_transient,FPS)//FPS).astype(int))

                tr_idx+=1
            plt.suptitle(pathway+'  ; Promimnence : '+str(prom)+' ; $\\theta$$_{bout}$ = '+str(bout_theta)+
                         's\n'+behaviors[b_cur]+' $\\rightarrow$ '+behaviors[b_next])
            plt.subplots_adjust(left = 0.05 , right = 0.99,top = 0.85)
            plt.savefig(output_folder+c+'/TimeToTransients/'+behaviors[b_cur]+'_'+behaviors[b_next]+'.png',dpi=300)
            plt.close()






#%% Heatmaps and means of behavior related transients (BRTs) separated based on bout length  - 1-2,2-3,3-4,4-5,>5 - z scored over bout - V2
sig_type = 'z_over_bout'
from matplotlib.collections import LineCollection
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/binned_durations/Summary'
times = [1,2,3,4,5]
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
all_events = {}
all_events_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds
lims = {'df':{'drd1':[-0.05,0.05],'a2a':[-0.01,0.01]},'z':{'drd1':[-.7,2],'a2a':[-.7,2]},'z_over_bout':{'drd1':[-.7,2],'a2a':[-.7,2]}}
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    all_events[pathway] = {}
    all_events_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        all_events[pathway][b] = {}
        all_events_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={1:[],2:[],3:[],4:[],5:[]}
            all_events[pathway][b][tr] = []
            all_events_length[pathway][b][tr] = []

for c in CMT:
    for m in CMT[c]:
        if c in ['drd1_hm4di', 'drd1_hm3dq']:
            pathway = 'drd1'
        elif c in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
            pathway = 'a2a'
        elif c == 'controls':
            if m.startswith('cA'):
                pathway = 'a2a'
            else:
                pathway = 'drd1'
        else:
            break
        for tr in CMT[c][m]:
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']

                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    session_bouts = hf.groupedByDuration_bouts(predictions, b ,times)
                    for dur in times:
                        if dur!=times[-1]: window_end = (dur+1+2)*seconds
                        else: window_end = (dur+2)*seconds
                        window_start = -2*seconds
                        for bout_idx in range(len(session_bouts[dur])):
                            if 'saline' in tr: tr = 'saline'
                            if side== 'left':
                                if session_bouts[dur][bout_idx][0]>=2*seconds and left.shape[0] >= session_bouts[dur][bout_idx][-1] + 2*seconds:
                                    sig_left_tot = left[session_bouts[dur][bout_idx][0] +window_start:session_bouts[dur][bout_idx][0]+window_end]
                                    z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                    photometry[pathway][b][tr][dur].append(z_over_bout_left)
                                    all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                    all_events[pathway][b][tr].append(stats.zscore(left[session_bouts[dur][bout_idx][0] - 2*seconds:session_bouts[dur][bout_idx][0] + 7*seconds],nan_policy='omit'))
                            elif side == 'right':
                                if session_bouts[dur][bout_idx][0]>=2*seconds and right.shape[0] >= session_bouts[dur][bout_idx][-1] + 2*seconds:
                                    sig_right_tot = right[session_bouts[dur][bout_idx][0] +window_start:session_bouts[dur][bout_idx][0]+window_end]
                                    z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                    bout_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                    photometry[pathway][b][tr][dur].append(z_over_bout_right)

                                    all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                    all_events[pathway][b][tr].append(stats.zscore(right[session_bouts[dur][bout_idx][0] - 2*seconds:session_bouts[dur][bout_idx][0] + 7*seconds],nan_policy='omit'))
                            else:
                                if session_bouts[dur][bout_idx][0]>=2*seconds and left.shape[0] >= session_bouts[dur][bout_idx][-1] + 2*seconds:
                                    sig_left_tot = left[session_bouts[dur][bout_idx][0]+window_start:session_bouts[dur][bout_idx][0]+window_end]
                                    z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                    photometry[pathway][b][tr][dur].append(z_over_bout_left)

                                    sig_right_tot = right[session_bouts[dur][bout_idx][0]+window_start:session_bouts[dur][bout_idx][0]+window_end]
                                    z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                    photometry[pathway][b][tr][dur].append(z_over_bout_right)

                                    all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                    all_events[pathway][b][tr].append(stats.zscore(left[session_bouts[dur][bout_idx][0] - 2*seconds:session_bouts[dur][bout_idx][0] + 7*seconds],nan_policy='omit'))
                                    all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                    all_events[pathway][b][tr].append(stats.zscore(right[session_bouts[dur][bout_idx][0] - 2 * seconds:session_bouts[dur][bout_idx][0] + 7 * seconds],nan_policy='omit'))

cohorts = ['drd1','a2a']
labels = ['${{1s} \geq {|event|<2s}}$','${{2s} \geq {|event|<3s}}$','${{3s} \geq {|event|<4s}}$','${{4s} \geq {|event|<5s}}$','${{5s} \geq {|event|}}$']
dur_colors = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51']
behaviors_prime = [0,1,2,3,4,6,7]
print('start plotting')
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=2, ncols=7, figsize=(20,5))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            if len(all_events_length[c][b][tr])==0:
                tr_idx+=1
                continue
            dur_idx=0
            print(b,c)
            for dur in times[:-1]:
                print(b,c,tr, dur,len(photometry[c][b][tr][dur]))
                if len(photometry[c][b][tr][dur])>0:
                    transients = np.array(photometry[c][b][tr][dur])
                    print(transients.shape)
                    num_of_transients = transients.shape[0]
                    plt.sca(ax[0,tr_idx])
                    mean = np.nanmean(transients,axis=0)
                    stderr = np.sqrt(np.nanvar(transients, axis=0)/num_of_transients)
                    plt.plot(mean, label=labels[dur_idx]+',$(N='+str(num_of_transients)+')$',c=dur_colors[dur_idx], alpha=line_alpha)
                    plt.fill_between(np.arange((dur+5)*seconds), y1=mean - stderr,y2=mean + stderr, color=dur_colors[dur_idx], alpha=low_alpha)
                    # plt.legend()
                    plt.xticks(np.arange((4+dur+1)*seconds,1*seconds),(np.arange(-2*seconds , (dur+1+2)*seconds,1*seconds)/seconds).astype(int))
                    plt.xlabel('Time[s]')
                    plt.ylabel('${\Delta F/F (Z)}$')
                    plt.vlines(x=(dur+2)*seconds, ymin=lims[sig_type][c][0], ymax=lims[sig_type][c][1], ls='dashed', colors='k')
                dur_idx+=1

            transients = np.array(photometry[c][b][tr][times[-1]])
            num_of_transients = transients.shape[0]
            if num_of_transients>0:
                plt.sca(ax[0, tr_idx])
                start_mean = np.nanmean(transients[:,:4*seconds], axis=0)
                start_stderr = np.sqrt(np.nanvar(transients[:,:4*seconds], axis=0) / num_of_transients)
                end_mean = np.nanmean(transients[:,-4*seconds:], axis=0)
                end_stderr = np.sqrt(np.nanvar(transients[:,-4*seconds:], axis=0) / num_of_transients)
                plt.plot(np.arange(4*seconds) , start_mean, label=labels[dur_idx] + ',$(N=' + str(num_of_transients) + ')$', c=dur_colors[dur_idx], alpha=line_alpha)
                plt.fill_between(np.arange(4*seconds), y1=start_mean - start_stderr, y2=start_mean + start_stderr, color=c1, alpha=low_alpha)
                plt.plot(np.arange(11 * seconds,15*seconds,1), end_mean,  c=dur_colors[dur_idx],alpha=line_alpha)
                plt.fill_between(np.arange(11 * seconds,15*seconds,1), y1 = end_mean - end_stderr, y2=end_mean + end_stderr, color=c1, alpha=low_alpha)
                # plt.legend()
                plt.xlabel('Time[s]')
                plt.ylabel('${\Delta F/F (Z)}$')
                plt.ylim(lims[sig_type][c])
                # plt.vlines(x=2*seconds, ymin=lims[sig_type][c][0],ymax=lims[sig_type][c][1], ls='dashed', colors='k')
                plt.vlines(x=13 * seconds, ymin=lims[sig_type][c][0], ymax=lims[sig_type][c][1], ls='dashed', colors='k')
            dur_idx += 1
            plt.vlines(x=2 * seconds, ymin=lims[sig_type][c][0], ymax=lims[sig_type][c][1], ls='dashed', colors='k')
            plt.xticks(np.arange(0,(15 * seconds)+1,1*seconds ), [-2, -1, 0, 1, 2,3,4,5,6,7, '', -2, -1, 0, 1, 2])
            all_events_mat = np.array(all_events[c][b][tr])
            events_length = np.array(all_events_length[c][b][tr])
            indices = np.argsort(events_length)
            ax[1, tr_idx] = sns.heatmap(all_events_mat[indices,], vmin= lims[sig_type][c][0], vmax=lims[sig_type][c][1], ax=ax[1, tr_idx],cbar=False)
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[0,tr_idx])
            colors = ["k", "white", "k"]
            x = [0, 10*seconds, 11*seconds, 15*seconds]
            y = [0, 0, 0, 0]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors, linewidth=1,transform=ax[0,tr_idx].get_xaxis_transform(), clip_on=False)
            ax[0,tr_idx].add_collection(lc)
            ax[0,tr_idx].spines["bottom"].set_visible(False)
            plt.sca(ax[1,tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, 9 * seconds,1*seconds),((np.arange(0, 9*seconds, 2 * seconds) - 2 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])
            plt.title(tr)
            tr_idx+=1

        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.01)
        plt.suptitle(c+';  $\\theta$$_{short}^{long}$ = 2s ; Quiet BL period = 3s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/'+ c + '/'+behaviors[b]+'_heatmapsAndMeans_'+sig_type+'V2.png',dpi=300)
        plt.close()