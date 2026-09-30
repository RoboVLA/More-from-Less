'use strict';
document.querySelectorAll('[role="tab"]').forEach(button=>{
  button.addEventListener('click',()=>{
    const group=button.dataset.group;
    document.querySelectorAll('[role="tab"]').forEach(other=>{
      if(other.dataset.group===group){other.setAttribute('aria-selected',String(other===button));other.tabIndex=other===button?0:-1;}
    });
    document.querySelectorAll('[role="tabpanel"]').forEach(panel=>{if(panel.dataset.group===group)panel.hidden=panel.id!==button.dataset.panel;});
  });
  button.addEventListener('keydown',event=>{
    if(event.ctrlKey || event.metaKey || event.altKey)return;
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    const tabs=[...document.querySelectorAll('[role="tab"]')].filter(t=>t.dataset.group===button.dataset.group);
    const i=tabs.indexOf(button);let n=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    event.preventDefault();tabs[n].click();tabs[n].focus();
  });
});

const demoVideos=[...document.querySelectorAll('.teaser-videos video')];
const demoToggle=document.getElementById('toggle-demos');
if(demoToggle && demoVideos.length){
  const updateDemoLabel=()=>{demoToggle.textContent=demoVideos.some(v=>!v.paused)?'Pause demos':'Play demos';};
  demoToggle.addEventListener('click',async()=>{
    if(demoVideos.some(v=>!v.paused)){demoVideos.forEach(v=>v.pause());}
    else{await Promise.allSettled(demoVideos.map(v=>v.play()));}
    updateDemoLabel();
  });
  demoVideos.forEach(v=>['play','pause','ended'].forEach(event=>v.addEventListener(event,updateDemoLabel)));
}
document.getElementById('copy-citation')?.addEventListener('click',async()=>{
  const text=document.getElementById('bibtex').textContent;
  const status=document.getElementById('copy-status');
  try{await navigator.clipboard.writeText(text);status.textContent='BibTeX copied.';}catch{status.textContent='Select and copy the citation above.';}
});
if(document.getElementById('file-content')){
  const links=[...document.querySelectorAll('[data-source]')];
  let requestNumber=0;
  async function show(path){
    if(!links.some(a=>a.dataset.source===path))return;
    const request=++requestNumber;
    const content=document.getElementById('file-content');
    const status=document.getElementById('source-status');
    document.getElementById('source-title').textContent=path;
    document.getElementById('source-download').href=path;
    document.getElementById('source-github').href='https://github.com/RoboVLA/More-from-Less/blob/main/'+path;
    links.forEach(a=>{if(a.dataset.source===path)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');});
    content.textContent='Loading source…';status.textContent='';
    try{
      const response=await fetch(path);
      if(!response.ok)throw Error('HTTP '+response.status);
      const text=await response.text();
      if(request!==requestNumber)return;
      content.textContent=text;
    }catch(error){
      if(request!==requestNumber)return;
      content.textContent='';
      status.textContent=location.protocol==='file:'
        ? 'Use an HTTP preview server: python scripts/serve.py'
        : 'This file could not be loaded. Retry, or use View on GitHub. ('+error.message+')';
    }
  }
  links.forEach(a=>a.addEventListener('click',e=>{
    if(e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;
    e.preventDefault();history.replaceState(null,'','#'+encodeURIComponent(a.dataset.source));show(a.dataset.source);
  }));
  function selectedPath(){try{return decodeURIComponent(location.hash.slice(1));}catch{return '';}}
  window.addEventListener('hashchange',()=>show(selectedPath()));
  const initial=selectedPath();show(links.some(a=>a.dataset.source===initial)?initial:'README.md');
}
