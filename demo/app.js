const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const menu=$('.menu');menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',String(open));menu.setAttribute('aria-label',open?'关闭导航':'展开导航');$('#navigation').classList.toggle('open',open)});
$$('nav a').forEach(a=>a.addEventListener('click',()=>{menu.setAttribute('aria-expanded','false');$('#navigation').classList.remove('open')}));
if('IntersectionObserver'in window){const reveal=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.remove('waiting');reveal.unobserve(e.target)}}),{threshold:.1});if(!reduced)$$('.reveal').filter(el=>!el.closest('.chapter-content')).forEach(el=>{el.classList.add('waiting');reveal.observe(el)});const active=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting)$$('nav a').forEach(a=>{const current=a.hash==='#'+e.target.id;a.classList.toggle('active',current);if(current)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current')})}),{rootMargin:'-15% 0px -65% 0px'});$$('main>section').forEach(el=>active.observe(el))}
function tabs(selector,callback){const buttons=$$(selector);buttons.forEach(b=>b.addEventListener('click',()=>{buttons.forEach(x=>{x.setAttribute('aria-selected',String(x===b));x.tabIndex=x===b?0:-1});callback(b)}));buttons.forEach((b,i)=>{b.tabIndex=b.getAttribute('aria-selected')==='true'?0:-1;b.addEventListener('keydown',e=>{let next;if(e.key==='ArrowRight')next=(i+1)%buttons.length;if(e.key==='ArrowLeft')next=(i+buttons.length-1)%buttons.length;if(e.key==='Home')next=0;if(e.key==='End')next=buttons.length-1;if(next!==undefined){e.preventDefault();buttons[next].focus();buttons[next].click()}})})}
const models={
base:{badge:'N0 / 基础模型',description:'仅使用正常组训练数据，以硬标签监督学习运动意图。',shape:'62 通道 × 1000 点',note:'模型一不启用知识蒸馏与人群混合；保留时序与 CSP 空间特征、数据增强和方差正则。'},
mixed:{badge:'M1 / 混合蒸馏',description:'正常组与 BCI 盲组混合采样，教师软目标辅助学生学习。',shape:'62 通道 × 1000 点',note:'综合报告参数说明：硬标签监督、温度 KL 蒸馏与方差正则；λ = 0.5625 表示正常组采样占比。推理保留学生与固定预处理。'},
sdad:{badge:'SDAD-Net / 分阶段域对抗蒸馏',description:'先形成分类能力，再以高低组均衡批次进行后期域适配。',shape:'11 通道 × 1000 点',note:'实线为学生推理路径，虚线为训练辅助路径。标准化、CSP 与特征缩放仅在训练数据上拟合。'}
};
function renderModel(model){const d=models[model];$('#base-details').hidden=model!=='base';$('#shared-details').hidden=model==='sdad';$('#training-details').hidden=model!=='mixed';$('#sdad-details').hidden=model!=='sdad';$('#model-badge').textContent=d.badge;$('#model-description').textContent=d.description;document.querySelectorAll('.model-diagram').forEach(figure=>{figure.hidden=figure.id!=='flow-'+model});$('#model-note').textContent=d.note}
tabs('[data-model]',b=>{$('#model-panel').setAttribute('aria-labelledby',b.id);renderModel(b.dataset.model)});renderModel('sdad');
const modes={offline:{title:'回放离线模型，预测小车动作',copy:'加载训练完成的模型与脑电记录，逐试次显示波形、运动意图与预测置信度。',prefix:'trial',label:'01',steps:[['加载模型与记录','选择学生模型、CSP、标准化器和脑电记录。'],['回放与识别','逐试次展示 C3、Cz、C4 波形，读取运动意图。'],['预测小车动作','将识别结果映射为小车动作，观察实物表现。']]},network:{title:'在线接收脑电，持续预测小车行为',copy:'通过终端启动识别界面，在在线模式下接收脑电数据，显示波形并生成动作预测。',prefix:'network',label:'02',steps:[['终端启动','输入启动命令，打开脑电识别 GUI。'],['在线接收','选择网络接收，设置数据源地址和端口。'],['行为预测','对接收的脑电进行推理，观察小车行为预测。']]},control:{title:'获取脑电、显示波形、识别并控制',copy:'实时脑电进入处理链路，波形加载后输出运动意图，将预测结果转换为小车动作。',prefix:'network',label:'03',steps:[['实时获取脑电','在线获取脑电信号，完成通道对齐与信号处理。'],['波形与预测','加载脑电波形，输出左右手意图及置信度。'],['驱动小车','由预测结果生成控制指令，执行小车动作。']]}};
let mode='offline',frame=0;const video=$('#car-video');
function showFrame(value){frame=(value+12)%12;const path=`assets/${modes[mode].prefix}-${String(frame).padStart(2,'0')}.png`;$('#gui-image').src=path;$('#gui-image').alt=modes[mode].title+'，GUI 记录 '+(frame+1);$('#playback-label').textContent='GUI 记录 / '+String(frame+1).padStart(2,'0')}
function renderMode(){const d=modes[mode];$('#demo-title').textContent=d.title;$('#demo-copy').textContent=d.copy;$('#video-label').textContent=d.label;$('#terminal-panel').hidden=mode!=='network';$('#demo-steps').replaceChildren(...d.steps.map(([title,copy])=>{const li=document.createElement('li'),div=document.createElement('div'),strong=document.createElement('strong'),p=document.createElement('p');strong.textContent=title;p.textContent=copy;div.append(strong,p);li.append(div);return li}));showFrame(0)}
tabs('[data-demo]',b=>{video.pause();mode=b.dataset.demo;video.src=`assets/${mode}.mp4`;video.load();$('#demo-panel').setAttribute('aria-labelledby',b.id);renderMode()});renderMode();
$('#previous').addEventListener('click',()=>showFrame(frame-1));$('#next').addEventListener('click',()=>showFrame(frame+1));
video.addEventListener('timeupdate',()=>{if(Number.isFinite(video.duration)&&video.duration>0){const n=Math.min(11,Math.floor(video.currentTime/video.duration*12));if(n!==frame)showFrame(n)}});
$('#copy-command').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('#terminal-command').textContent);$('#copy-status').textContent='命令已复制。'}catch{$('#copy-status').textContent='请选中上方命令并复制。'}});
const dialog=$('#image-dialog');$('#enlarge').addEventListener('click',()=>{dialog.querySelector('img').src=$('#gui-image').src;dialog.showModal()});$('#close-dialog').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close()});

// The background is decorative: it never receives clicks or blocks page controls.
const motionPreference=matchMedia('(prefers-reduced-motion: reduce)');
const ambient=$('#ambient-shapes'),ctx=ambient.getContext('2d');
const pointer={x:-1000,y:-1000}; let shapes=[], viewWidth=0, viewHeight=0, animation=0, lastFrame=0;
const random=(min,max)=>min+Math.random()*(max-min);
function resizeAmbient(){
  const oldWidth=viewWidth,oldHeight=viewHeight;
  viewWidth=innerWidth;viewHeight=innerHeight;
  const dpr=Math.min(devicePixelRatio||1,2);
  ambient.width=Math.round(viewWidth*dpr);ambient.height=Math.round(viewHeight*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  if(!shapes.length){
    const count=Math.max(12,Math.min(34,Math.round(viewWidth*viewHeight/42000)));
    shapes=Array.from({length:count},()=>({x:random(0,viewWidth),y:random(0,viewHeight),w:random(28,116),h:random(24,100),vx:random(-.4,.4),vy:random(-.35,.35),driftX:random(-.4,.4),driftY:random(-.35,.35),driftTimer:random(120,360),angle:random(-.7,.7),spin:random(-.002,.002),radius:0,filled:false,alpha:random(.12,.27),scaleX:1,scaleY:1,cool:0}));
  }else if(oldWidth&&oldHeight){shapes.forEach(s=>{s.x=s.x/oldWidth*viewWidth;s.y=s.y/oldHeight*viewHeight})}
}
function renderAmbient(now){
  const dt=lastFrame?Math.min((now-lastFrame)/16.67,2):1;lastFrame=now;
  ctx.clearRect(0,0,viewWidth,viewHeight);
  for(const s of shapes){
    const dx=s.x-pointer.x,dy=s.y-pointer.y,dist=Math.hypot(dx,dy),reach=Math.max(s.w,s.h)*.55+25;
    if(!motionPreference.matches&&dist<reach&&s.cool<=0){
      const inv=1/Math.max(dist,1),push=random(1.5,2.6);
      s.vx+=dx*inv*push;s.vy+=dy*inv*push;s.spin=random(-.025,.025);s.scaleX=random(.65,1.4);s.scaleY=random(.65,1.4);s.radius=0;s.cool=24;
    }
    if(!motionPreference.matches){
      s.x+=s.vx*dt;s.y+=s.vy*dt;s.angle+=s.spin*dt;s.cool-=dt;
      s.driftTimer=(s.driftTimer??0)-dt;
      if(s.driftTimer<=0){const angle=random(0,Math.PI*2),speed=random(.18,.46);s.driftX=Math.cos(angle)*speed;s.driftY=Math.sin(angle)*speed;s.driftTimer=random(120,360)}
      s.vx+=((s.driftX??0)-s.vx)*.012*dt;s.vy+=((s.driftY??0)-s.vy)*.012*dt;s.spin*=Math.pow(.98,dt);
      s.scaleX+=(1-s.scaleX)*.015*dt;s.scaleY+=(1-s.scaleY)*.015*dt;
      const pad=140;if(s.x<-pad)s.x=viewWidth+pad;if(s.x>viewWidth+pad)s.x=-pad;if(s.y<-pad)s.y=viewHeight+pad;if(s.y>viewHeight+pad)s.y=-pad;
    }
    ctx.save();ctx.translate(s.x,s.y);ctx.rotate(s.angle);ctx.scale(s.scaleX,s.scaleY);ctx.globalAlpha=s.alpha;
    ctx.fillStyle='#25857c';ctx.strokeStyle='#65b9a8';ctx.lineWidth=1;
    ctx.beginPath();ctx.rect(-s.w/2,-s.h/2,s.w,s.h);ctx.stroke();ctx.restore();
  }
  if(!motionPreference.matches&&!document.hidden)animation=requestAnimationFrame(renderAmbient);
}
function startAmbient(){cancelAnimationFrame(animation);lastFrame=0;resizeAmbient();renderAmbient(0)}
resizeAmbient();renderAmbient(0);
window.addEventListener('resize',()=>{resizeAmbient();if(motionPreference.matches)renderAmbient(0)},{passive:true});
window.addEventListener('pointermove',e=>{pointer.x=e.clientX;pointer.y=e.clientY},{passive:true});
window.addEventListener('pointerout',e=>{if(!e.relatedTarget){pointer.x=-1000;pointer.y=-1000}},{passive:true});
document.addEventListener('visibilitychange',()=>{if(document.hidden)cancelAnimationFrame(animation);else startAmbient()});
motionPreference.addEventListener('change',startAmbient);

// The supplied clip is an illustrative backdrop and is paused outside the viewport.
const conceptVideo=$('#concept-video');
if('IntersectionObserver'in window){
  let isVisible=false;
  const syncVideo=()=>{if(isVisible&&!motionPreference.matches&&!document.hidden)conceptVideo.play().catch(()=>{});else conceptVideo.pause()};
  new IntersectionObserver(entries=>{isVisible=entries[0].isIntersecting;syncVideo()},{threshold:.15}).observe(conceptVideo);
  motionPreference.addEventListener('change',syncVideo);document.addEventListener('visibilitychange',syncVideo);
}

// Reveal the corresponding accordion when navigating to a chapter.
function openChapter(id){const section=document.getElementById(id);const chapter=section?.matches('.chapter-accordion')?section:section?.querySelector(':scope > .chapter-accordion');if(chapter)chapter.open=true;}
document.querySelectorAll('a[href^="#"]').forEach(link=>link.addEventListener('click',()=>openChapter(link.getAttribute('href').slice(1))));
if(location.hash)openChapter(location.hash.slice(1));
window.addEventListener('hashchange',()=>openChapter(location.hash.slice(1)));

// Keep the cover readable without scripting; add a restrained scroll response when motion is allowed.
const cover=document.querySelector('#hero-cover');let coverFrame=0;
function updateCover(){coverFrame=0;if(!cover||motionPreference.matches)return;const distance=Math.max(0,Math.min(1,scrollY/Math.max(cover.offsetHeight,1)));cover.style.setProperty('--cover-shift',`${Math.round(distance*48)}px`);cover.style.setProperty('--cover-opacity',String(1-distance*.28));}
window.addEventListener('scroll',()=>{if(!coverFrame&&!motionPreference.matches)coverFrame=requestAnimationFrame(updateCover)},{passive:true});
if(cover)updateCover();
motionPreference.addEventListener('change',()=>{if(motionPreference.matches){cover.style.removeProperty('--cover-shift');cover.style.removeProperty('--cover-opacity')}else updateCover()});
const coverWave=document.querySelector('.hero-wave');let coverVisible=true;
function syncCoverWave(){if(coverWave)coverWave.classList.toggle('is-paused',document.hidden||!coverVisible)}
if(coverWave&&'IntersectionObserver'in window){new IntersectionObserver(entries=>{coverVisible=entries[0].isIntersecting;syncCoverWave()},{threshold:0}).observe(coverWave)}
document.addEventListener('visibilitychange',syncCoverWave);

// Replay a measured reading-order entrance whenever a chapter or inner drawer opens.
const chapterAnimations=new Map();
const componentGroups='.section-heading,.problem-row,.metrics,.results-grid,.medical-grid,.drawer-deck,.application-feature,.prospect-roadmap,.figure-grid';
function stopChapterEntrance(details){
  (chapterAnimations.get(details)||[]).forEach(animation=>animation.cancel());
  chapterAnimations.delete(details);
}
function revealChapterComponents(details){
  stopChapterEntrance(details);
  if(!details.open||motionPreference.matches)return;
  const content=details.querySelector(':scope > .chapter-content, :scope > .drawer-body');
  if(!content)return;
  const components=[...content.children].flatMap(el=>el.matches(componentGroups)?[...el.children]:[el])
    .filter(el=>!el.hidden&&el.getClientRects().length);
  const animations=components.map((el,index)=>el.animate([
    {opacity:0,transform:'translateY(24px)'},
    {opacity:1,transform:'translateY(0)'}
  ],{duration:900,delay:Math.min(index*180,1800),easing:'cubic-bezier(.22,1,.36,1)',fill:'backwards'}));
  chapterAnimations.set(details,animations);
  Promise.all(animations.map(animation=>animation.finished.catch(()=>{}))).then(()=>{
    if(chapterAnimations.get(details)===animations)stopChapterEntrance(details);
  });
}
document.querySelectorAll('.chapter-accordion,.drawer-card').forEach(details=>{
  details.addEventListener('toggle',()=>{
    if(!details.open){
      stopChapterEntrance(details);
      details.querySelectorAll('details').forEach(stopChapterEntrance);
    }else revealChapterComponents(details);
  });
  if(details.open)revealChapterComponents(details);
});
motionPreference.addEventListener('change',()=>{
  if(motionPreference.matches)[...chapterAnimations.keys()].forEach(stopChapterEntrance);
});
