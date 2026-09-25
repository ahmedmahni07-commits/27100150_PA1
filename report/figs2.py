import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R='/home/claude/rep/res/'
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']
plt.rcParams.update({'font.size':7,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.25,'axes.axisbelow':True,'font.family':'serif'})
CL=['dog','elephant','giraffe','guitar','horse','house','person']
J=lambda p: json.load(open(R+p))
kw=dict(marker='o',ms=2,lw=1.2)
# ---------------- Task 2
H={m:J(f'task2/results/{m}_results.json')['history'] for m in ['source_only','dan','dann','cdan']}
pre=J('task2/results/pre_stabilization/dann_results.json')['history']
ev=J('task2/results/evaluate_final_results.json')
N={'source_only':'Source-only','dan':'DAN','dann':'DANN','cdan':'CDAN'}
ep=lambda h:[e['epoch'] for e in h]
fig,ax=plt.subplots(1,4,figsize=(7.4,1.75))
for k,(m,h) in enumerate(H.items()):
    ax[0].plot(ep(h),[e['mean_val_macro_f1'] for e in h],color=C[k],label=N[m],**kw)
ax[0].set_ylim(0.80,0.96); ax[0].set_ylabel('src-val macro-F1'); ax[0].legend(fontsize=5,frameon=False,ncol=2,loc='lower center')
ax[0].set_title('(a) selection signal',fontsize=7)
ax[1].plot(ep(H['source_only']),[e['train_loss'] for e in H['source_only']],color=C[0],label='Src-only CE',**kw)
ax[1].plot(ep(H['dan']),[e['train_loss'] for e in H['dan']],color=C[1],label='DAN CE+MMD',**kw)
for k,m in [(2,'dann'),(3,'cdan')]:
    ax[1].plot(ep(H[m]),[e['train_cls_loss'] for e in H[m]],color=C[k],label=f'{N[m]} CE',**kw)
    ax[1].plot(ep(H[m]),[e['train_domain_loss'] for e in H[m]],color=C[k],ls='--',lw=1.0,label=f'{N[m]} dom.')
ax[1].set_ylabel('train loss'); ax[1].legend(fontsize=4.6,frameon=False,ncol=2,loc='upper right'); ax[1].set_ylim(0,1.65)
ax[1].set_title('(b) training losses',fontsize=7)
for k,m in [(2,'dann'),(3,'cdan')]:
    ax[2].plot(ep(H[m]),[e['train_domain_acc'] for e in H[m]],color=C[k],label=f'{N[m]} (clipped)',**kw)
ax[2].plot(ep(pre),[e['train_domain_acc'] for e in pre],ls='--',lw=1.0,color='#888',label='DANN (unclipped)')
ax[2].axhline(0.5,color='#444',lw=0.7,ls=':'); ax[2].set_ylim(0.3,0.9)
ax[2].set_ylabel('discriminator acc.'); ax[2].legend(fontsize=5,frameon=False)
ax[2].set_title('(c) domain discriminator',fontsize=7)
for a in ax[:3]: a.set_xlabel('epoch')
w=0.26; x=np.arange(7)
for k,m in enumerate(['dan','dann','cdan']):
    d=np.array(ev[m]['per_class_comparison_vs_source_only']['per_class_delta'])*100
    ax[3].bar(x+(k-1)*w,d,w*0.9,color=C[k+1],label=N[m])
ax[3].axhline(0,color='#444',lw=0.7); ax[3].set_xticks(x); ax[3].set_xticklabels(CL,rotation=45,ha='right',fontsize=5.5)
ax[3].set_ylabel('Δ Sketch acc. (pp)'); ax[3].legend(fontsize=5,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.03),handlelength=1,columnspacing=0.6)
ax[3].set_title('(d) per-class transfer',fontsize=7,pad=9)
plt.tight_layout(pad=0.3,w_pad=0.6); plt.savefig('t2.pdf'); plt.close()
# ---------------- Task 3
fig,ax=plt.subplots(1,4,figsize=(7.4,1.75))
erm=H['source_only']; sam=J('task3/results/sam_results.json')['history']
dd={lab:J(f'task3/results/{f}_results.json')['history'] for f,lab in [('dan_dg_lambda01','DAN-DG λ=0.1'),('dan_dg','DAN-DG λ=1'),('dan_dg_lambda10','DAN-DG λ=10')]}
ax[0].plot(ep(erm),[e['train_loss'] for e in erm],color='#444',label='ERM',**kw)
ax[0].plot(ep(sam),[e['train_cls_loss'] for e in sam],color=C[3],label='SAM',**kw)
for k,(lab,h) in enumerate(dd.items()):
    ax[0].plot(ep(h),[e['train_cls_loss'] for e in h],color=C[k],label=lab,**kw)
ax[0].axhline(np.log(7),color='#444',lw=0.7,ls=':'); ax[0].set_ylabel('train CE'); ax[0].legend(fontsize=5,frameon=False,loc='center right')
ax[0].set_title('(a) classification loss',fontsize=7)
for k,(lab,h) in enumerate(dd.items()):
    ax[1].plot(ep(h),[e['train_mmd_loss'] for e in h],color=C[k],label=lab,**kw)
ax[1].set_ylabel('pairwise MMD²'); ax[1].set_title('(b) DAN-DG MMD penalty',fontsize=7)
ax[2].plot(ep(erm),[e['mean_val_macro_f1'] for e in erm],color='#444',label='ERM',**kw)
ax[2].plot(ep(sam),[e['mean_val_macro_f1'] for e in sam],color=C[3],label='SAM',**kw)
ax[2].plot(ep(dd['DAN-DG λ=0.1']),[e['mean_val_macro_f1'] for e in dd['DAN-DG λ=0.1']],color=C[0],label='DAN-DG λ=0.1',**kw)
ax[2].set_ylim(0.75,0.97); ax[2].set_ylabel('src-val macro-F1'); ax[2].legend(fontsize=5,frameon=False,loc='lower left')
ax[2].set_title('(c) selection signal',fontsize=7)
for a in ax[:3]: a.set_xlabel('epoch')
ev3=J('task3/results/evaluate_sketch_results.json'); cs=J('task3/results/dan_dg_controlled_study_results.json')
e0=np.array(ev3['erm']['per_class_sketch_accuracy'])
S=[('SAM',np.array(ev3['sam']['per_class_sketch_accuracy'])-e0,C[3]),('DAN-DG λ=0.1',np.array(cs['dan_dg_lambda01']['per_class_sketch_accuracy'])-e0,C[0]),('DAN (T2)',np.array(ev['dan']['per_class_target_accuracy'])-e0,C[1])]
for k,(lab,d,c) in enumerate(S): ax[3].bar(x+(k-1)*w,d*100,w*0.9,color=c,label=lab)
ax[3].axhline(0,color='#444',lw=0.7); ax[3].set_xticks(x); ax[3].set_xticklabels(CL,rotation=45,ha='right',fontsize=5.5)
ax[3].set_ylabel('Δ vs ERM (pp)'); ax[3].legend(fontsize=5,frameon=False,loc='lower left',handlelength=1)
ax[3].set_title('(d) per-class change vs ERM',fontsize=7)
plt.tight_layout(pad=0.3,w_pad=0.6); plt.savefig('t3.pdf'); plt.close(); print('ok')
