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
  async function show(path){
    if(!links.some(a=>a.dataset.source===path))return;
    document.getElementById('source-title').textContent=path;
    const download=document.getElementById('source-download');download.href=path;
    try{const response=await fetch(path);if(!response.ok)throw Error(response.status);document.getElementById('file-content').textContent=await response.text();}
    catch{document.getElementById('file-content').textContent='Start the local server to browse source files: python scripts/serve.py';}
  }
  links.forEach(a=>a.addEventListener('click',e=>{e.preventDefault();show(a.dataset.source);}));
  show('README.md');
}
