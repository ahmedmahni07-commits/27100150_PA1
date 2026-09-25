import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='/home/claude/rep/res/'; O='/home/claude/rep/figs/'
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25,'axes.axisbelow':True,'font.family':'serif'})
CL=['dog','elephant','giraffe','guitar','horse','house','person']
J=lambda p: json.load(open(R+p))
# ---------------- Task 2
H={m:J(f'task2/results/{m}_results.json')['history'] for m in ['source_only','dan','dann','cdan']}
pre=J('task2/results/pre_stabilization/dann_results.json')['history']
ev=J('task2/results/evaluate_final_results.json')
fig,ax=plt.subplots(1,3,figsize=(7.2,1.85))
names={'source_only':'Source-only','dan':'DAN','dann':'DANN','cdan':'CDAN'}
for k,(m,h) in enumerate(H.items()):
    ax[0].plot([e['epoch'] for e in h],[e['mean_val_macro_f1'] for e in h],marker='o',ms=2.5,lw=1.4,color=C[k],label=names[m])
ax[0].set_ylim(0.80,0.96); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('mean src-val macro-F1'); ax[0].legend(fontsize=5.5,frameon=False,ncol=2,loc='lower center')
ax[0].set_title('(a) model-selection signal',fontsize=8)
for k,m in enumerate(['dann','cdan']):
    h=H[m]; ax[1].plot([e['epoch'] for e in h],[e['train_domain_acc'] for e in h],marker='o',ms=2.5,lw=1.4,color=C[k+2],label=f'{names[m]} (clipped)')
ax[1].plot([e['epoch'] for e in pre],[e['train_domain_acc'] for e in pre],ls='--',lw=1.2,color='#888',label='DANN (unclipped)')
ax[1].axhline(0.5,color='#444',lw=0.8,ls=':'); ax[1].set_ylim(0.3,0.9)
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('train discriminator acc.'); ax[1].legend(fontsize=6.5,frameon=False)
ax[1].set_title('(b) domain discriminator',fontsize=8)
w=0.26; x=np.arange(7)
for k,m in enumerate(['dan','dann','cdan']):
    d=np.array(ev[m]['per_class_comparison_vs_source_only']['per_class_delta'])*100
    ax[2].bar(x+(k-1)*w,d,w*0.9,color=C[k+1],label=names[m])
ax[2].axhline(0,color='#444',lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(CL,rotation=40,ha='right',fontsize=6.5)
ax[2].set_ylabel('Δ Sketch acc. (pp)'); ax[2].legend(fontsize=6,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.02),handlelength=1,columnspacing=0.8)
ax[2].set_title('(c) per-class transfer',fontsize=8,pad=10)
plt.tight_layout(); plt.savefig(O+'t2.pdf'); plt.close()
# ---------------- Task 3
fig,ax=plt.subplots(1,3,figsize=(7.2,1.85))
for k,(f,lab) in enumerate([('dan_dg_lambda01','λ=0.1'),('dan_dg','λ=1'),('dan_dg_lambda10','λ=10')]):
    h=J(f'task3/results/{f}_results.json')['history']
    ax[0].plot([e['epoch'] for e in h],[e['train_cls_loss'] for e in h],marker='o',ms=2.5,lw=1.4,color=C[k],label=lab)
    ax[1].plot([e['epoch'] for e in h],[e['train_mmd_loss'] for e in h],marker='o',ms=2.5,lw=1.4,color=C[k],label=lab)
ax[0].axhline(np.log(7),color='#444',lw=0.8,ls=':'); ax[0].text(9,np.log(7)+0.06,'ln 7 (chance)',fontsize=6.5)
ax[0].set_xlabel('epoch'); ax[0].set_ylabel('train CE'); ax[0].set_title('(a) DAN-DG classification loss',fontsize=8); ax[0].legend(fontsize=6.5,frameon=False)
ax[1].set_xlabel('epoch'); ax[1].set_ylabel('mean pairwise MMD²'); ax[1].set_title('(b) DAN-DG MMD penalty',fontsize=8)
ev3=J('task3/results/evaluate_sketch_results.json'); cs=J('task3/results/dan_dg_controlled_study_results.json')
erm=np.array(ev3['erm']['per_class_sketch_accuracy'])
series=[('SAM (DG)',np.array(ev3['sam']['per_class_sketch_accuracy'])-erm,C[0]),
        ('DAN-DG λ=0.1',np.array(cs['dan_dg_lambda01']['per_class_sketch_accuracy'])-erm,C[1]),
        ('DAN (UDA, T2)',np.array(ev['dan']['per_class_target_accuracy'])-erm,C[2])]
for k,(lab,d,c) in enumerate(series): ax[2].bar(x+(k-1)*w,d*100,w*0.9,color=c,label=lab)
ax[2].axhline(0,color='#444',lw=0.8); ax[2].set_xticks(x); ax[2].set_xticklabels(CL,rotation=40,ha='right',fontsize=6.5)
ax[2].set_ylabel('Δ vs ERM (pp)'); ax[2].legend(fontsize=5.8,frameon=False,loc='lower left',handlelength=1); ax[2].set_title('(c) per-class change vs ERM',fontsize=8)
plt.tight_layout(); plt.savefig(O+'t3.pdf'); plt.close()
print('ok')
