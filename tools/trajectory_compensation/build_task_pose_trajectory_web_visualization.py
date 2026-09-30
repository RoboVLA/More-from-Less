from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f3f6f9;
      --panel: #ffffff;
      --line: #d8e0e8;
      --text: #16202b;
      --muted: #657486;
      --blue: #2563eb;
      --green: #168a4a;
      --red: #dc2626;
      --amber: #d97706;
      --violet: #5b5bd6;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 "Segoe UI", "Microsoft YaHei", Arial, sans-serif; }
    .topbar { position: sticky; top: 0; z-index: 10; display: grid; grid-template-columns: minmax(260px,1fr) auto; gap: 14px; align-items: center; padding: 14px 18px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.94); }
    h1 { margin: 0; font-size: 18px; letter-spacing: 0; }
    .sub { margin-top: 3px; color: var(--muted); font-size: 12px; }
    .controls { display: grid; grid-template-columns: auto minmax(260px, 390px) 88px auto; gap: 10px; align-items: center; }
    button { width: 38px; height: 34px; border: 1px solid var(--line); border-radius: 6px; background: #fff; cursor: pointer; }
    input[type=range] { width: 100%; accent-color: var(--blue); }
    .readout { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
    .layout { display: grid; grid-template-columns: minmax(0,1.45fr) minmax(360px,.75fr); gap: 14px; padding: 14px; }
    .stack { display: grid; gap: 14px; align-content: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 8px 24px rgba(20,30,42,.07); }
    .head { min-height: 42px; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--line); background: #fbfcfe; }
    .head b { font-size: 13px; }
    .hint { color: var(--muted); font-size: 12px; }
    .scene { height: min(62vh, 670px); min-height: 420px; background: linear-gradient(#fbfdff, #edf3f8); }
    canvas { display: block; width: 100%; }
    #scene { height: 100%; cursor: grab; touch-action: none; }
    #scene:active { cursor: grabbing; }
    .chart { height: 246px; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; padding: 12px; }
    .metric { border: 1px solid var(--line); border-radius: 6px; padding: 8px 9px; background: #fff; min-width: 0; }
    .metric span { display: block; color: var(--muted); font-size: 11px; margin-bottom: 3px; }
    .metric strong { font-size: 18px; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
    .x strong { color: var(--blue); } .y strong { color: var(--green); } .z strong { color: var(--red); }
    .rx strong { color: var(--violet); } .ry strong { color: #0891b2; } .rz strong { color: var(--amber); }
    video { display: block; width: 100%; max-height: 280px; background: #111827; }
    .links { display: flex; gap: 8px; flex-wrap: wrap; padding: 10px 12px 12px; }
    .links a { color: var(--text); text-decoration: none; border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; font-size: 12px; background: #fff; }
    .table-wrap { max-height: 306px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
    th, td { padding: 7px 8px; text-align: right; border-bottom: 1px solid #e7edf4; white-space: nowrap; }
    th { position: sticky; top: 0; background: #f9fbfd; color: var(--muted); z-index: 1; }
    th:first-child, td:first-child { text-align: left; }
    tr.active { background: #eef5ff; }
    .note { padding: 9px 12px 12px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); background: #fbfcfe; }
    @media (max-width: 980px) { .topbar, .layout { grid-template-columns: 1fr; } .controls { grid-template-columns: auto minmax(0,1fr) 78px auto; } .metrics { grid-template-columns: repeat(2,minmax(0,1fr)); } .scene { min-height: 360px; height: 54vh; } }
  </style>
</head>
<body>
  <header class="topbar">
    <div><h1>__TITLE__</h1><div class="sub">Camera frame: +X right, +Y down, +Z forward. Position in mm, orientation in degrees.</div></div>
    <div class="controls">
      <button id="play" title="Play/Pause">▶</button>
      <input id="slider" type="range" min="0" value="0" step="1" />
      <div id="frameText" class="readout">0 / 0</div>
      <button id="reset" title="Reset view">⟲</button>
    </div>
  </header>
  <main class="layout">
    <section class="stack">
      <section class="panel"><div class="head"><b>3D pose trajectory</b><span class="hint">drag to rotate, wheel to zoom</span></div><div class="scene"><canvas id="scene"></canvas></div></section>
      <section class="panel"><div class="head"><b>Position curves</b><span class="hint">x / y / z</span></div><canvas id="xyz" class="chart"></canvas></section>
      <section class="panel"><div class="head"><b>Orientation curves</b><span class="hint">rx / ry / rz</span></div><canvas id="rpy" class="chart"></canvas></section>
    </section>
    <aside class="stack">
      <section class="panel">
        <div class="head"><b>Current 6D point</b><span id="timeText" class="hint">0.000 s</span></div>
        <div class="metrics">
          <div class="metric x"><span>x mm</span><strong id="xv">0.0</strong></div>
          <div class="metric y"><span>y mm</span><strong id="yv">0.0</strong></div>
          <div class="metric z"><span>z mm</span><strong id="zv">0.0</strong></div>
          <div class="metric rx"><span>rx deg</span><strong id="rxv">0.0</strong></div>
          <div class="metric ry"><span>ry deg</span><strong id="ryv">0.0</strong></div>
          <div class="metric rz"><span>rz deg</span><strong id="rzv">0.0</strong></div>
        </div>
        <div class="links">
          <a href="./moving_object_pose_xyzrpy_only.csv">xyzrpy CSV</a>
          <a href="./moving_object_pose_trajectory_xyzrpy.csv">full CSV</a>
          <a href="./moving_object_pose_trajectory_xyzrpy.json">JSON</a>
        </div>
      </section>
      <section class="panel"><div class="head"><b>Overlay video</b><span class="hint">tracking check</span></div><video src="./__OVERLAY_VIDEO__" controls muted playsinline></video></section>
      <section class="panel">
        <div class="head"><b>Trajectory table</b><span class="hint">frame / xyz / rpy</span></div>
        <div class="table-wrap"><table><thead><tr><th>frame</th><th>x</th><th>y</th><th>z</th><th>rx</th><th>ry</th><th>rz</th></tr></thead><tbody id="rows"></tbody></table></div>
        <div class="note">Orientation uses the visible moving-object long axis from RGB-D PCA. Rotation around that axis is constrained, not fully observed.</div>
      </section>
    </aside>
  </main>
  <script>
    const source = __DATA_JSON__;
    const data = source.trajectory.map((r) => ({ frame:+r.frame, time:+r.time_s, x:+r.x_mm, y:+r.y_mm, z:+r.z_mm, rx:+r.rx_deg, ry:+r.ry_deg, rz:+r.rz_deg, ax:+r.axis_x, ay:+r.axis_y, az:+r.axis_z }));
    const $ = (id) => document.getElementById(id);
    const state = { i:0, playing:false, yaw:-0.62, pitch:-0.36, zoom:1, drag:false, x:0, y:0, timer:null };
    const slider = $("slider"); slider.max = String(data.length - 1);
    const scene = $("scene"), xyz = $("xyz"), rpy = $("rpy");
    const fields = ["x","y","z"];
    const min = {}, max = {};
    for (const f of fields) { min[f] = Math.min(...data.map(p=>p[f])); max[f] = Math.max(...data.map(p=>p[f])); }
    const center = { x:(min.x+max.x)/2, y:(min.y+max.y)/2, z:(min.z+max.z)/2 };
    const extent = Math.max(max.x-min.x, max.y-min.y, max.z-min.z, 80);
    function fit(c) { const dpr = Math.max(devicePixelRatio||1,1), r=c.getBoundingClientRect(); c.width=Math.max(1,Math.round(r.width*dpr)); c.height=Math.max(1,Math.round(r.height*dpr)); const ctx=c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0); return {ctx,w:r.width,h:r.height}; }
    function world(p) { return {x:p.x-center.x, y:-(p.y-center.y), z:p.z-center.z}; }
    function rot(p) { const cy=Math.cos(state.yaw), sy=Math.sin(state.yaw), cp=Math.cos(state.pitch), sp=Math.sin(state.pitch); const x=p.x*cy+p.z*sy, z=-p.x*sy+p.z*cy; return {x, y:p.y*cp-z*sp, z:p.y*sp+z*cp}; }
    function proj(p,w,h) { const q=rot(p), s=Math.min(w,h)*0.54*state.zoom/extent; return {x:w*.5+q.x*s, y:h*.54-q.y*s, z:q.z}; }
    function line3(ctx,w,h,a,b,color,lw) { const p=proj(world(a),w,h), q=proj(world(b),w,h); ctx.strokeStyle=color; ctx.lineWidth=lw; ctx.lineCap="round"; ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); ctx.stroke(); }
    function drawScene() {
      const {ctx,w,h}=fit(scene); ctx.clearRect(0,0,w,h); ctx.fillStyle="#f8fbfd"; ctx.fillRect(0,0,w,h);
      const grid=Math.max(140, extent*1.6);
      for (let k=-4;k<=4;k++) { const t=k*grid/8; line3(ctx,w,h,{x:center.x-grid/2,y:center.y+t,z:center.z},{x:center.x+grid/2,y:center.y+t,z:center.z},"#dce5ee",1); line3(ctx,w,h,{x:center.x+t,y:center.y-grid/2,z:center.z},{x:center.x+t,y:center.y+grid/2,z:center.z},"#dce5ee",1); }
      line3(ctx,w,h,{x:center.x-grid/2,y:center.y,z:center.z},{x:center.x+grid/2,y:center.y,z:center.z},"#94a3b8",2);
      line3(ctx,w,h,{x:center.x,y:center.y-grid/2,z:center.z},{x:center.x,y:center.y+grid/2,z:center.z},"#94a3b8",2);
      for (let k=1;k<data.length;k++) line3(ctx,w,h,data[k-1],data[k],`hsl(${215-k/data.length*185} 78% 46%)`,4);
      const row=data[state.i], p=proj(world(row),w,h); ctx.fillStyle="#dc2626"; ctx.strokeStyle="#fff"; ctx.lineWidth=3; ctx.beginPath(); ctx.arc(p.x,p.y,7,0,Math.PI*2); ctx.fill(); ctx.stroke();
      const len=Math.max(34,extent*.28); line3(ctx,w,h,row,{x:row.x+row.ax*len,y:row.y+row.ay*len,z:row.z+row.az*len},"#dc2626",5);
      ctx.fillStyle="#657486"; ctx.font="12px Segoe UI"; ctx.fillText(`yaw ${state.yaw.toFixed(2)}  pitch ${state.pitch.toFixed(2)}  zoom ${state.zoom.toFixed(2)}`,14,h-16);
    }
    function range(fs) { const vals=fs.flatMap(f=>data.map(p=>p[f])); let a=Math.min(...vals), b=Math.max(...vals); if (Math.abs(b-a)<1e-6){a-=1;b+=1;} return [a,b]; }
    function chart(c, fs, colors, title) {
      const {ctx,w,h}=fit(c), pad={l:54,r:16,t:26,b:34}; ctx.clearRect(0,0,w,h); ctx.fillStyle="#fff"; ctx.fillRect(0,0,w,h); const [a,b]=range(fs), pw=w-pad.l-pad.r, ph=h-pad.t-pad.b;
      ctx.strokeStyle="#e3e9f0"; for(let n=0;n<=4;n++){const y=pad.t+ph*n/4; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke();}
      fs.forEach((f,j)=>{ctx.strokeStyle=colors[j]; ctx.lineWidth=2.4; ctx.beginPath(); data.forEach((row,i)=>{const x=pad.l+pw*i/(data.length-1), y=pad.t+ph-(row[f]-a)*ph/(b-a); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);}); ctx.stroke(); ctx.fillStyle=colors[j]; ctx.fillText(f,pad.l+j*70,h-12);});
      const x=pad.l+pw*state.i/(data.length-1); ctx.strokeStyle="#111827"; ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,h-pad.b); ctx.stroke(); ctx.fillStyle="#16202b"; ctx.font="700 13px Segoe UI"; ctx.fillText(title,pad.l,17); ctx.fillStyle="#657486"; ctx.font="12px Segoe UI"; ctx.fillText(`${a.toFixed(1)} .. ${b.toFixed(1)}`,w-128,17);
    }
    function update() {
      const r=data[state.i]; slider.value=String(state.i); $("frameText").textContent=`${r.frame} / ${data.length-1}`; $("timeText").textContent=`${r.time.toFixed(3)} s`;
      for (const [id,v] of [["xv",r.x],["yv",r.y],["zv",r.z],["rxv",r.rx],["ryv",r.ry],["rzv",r.rz]]) $(id).textContent=v.toFixed(1);
      document.querySelectorAll("tr.active").forEach(e=>e.classList.remove("active")); const tr=document.querySelector(`[data-i="${state.i}"]`); if(tr)tr.classList.add("active");
      drawScene(); chart(xyz,["x","y","z"],["#2563eb","#168a4a","#dc2626"],"position mm"); chart(rpy,["rx","ry","rz"],["#5b5bd6","#0891b2","#d97706"],"orientation deg");
    }
    $("rows").innerHTML = data.map((r,i)=>`<tr data-i="${i}"><td>${r.frame}</td><td>${r.x.toFixed(1)}</td><td>${r.y.toFixed(1)}</td><td>${r.z.toFixed(1)}</td><td>${r.rx.toFixed(1)}</td><td>${r.ry.toFixed(1)}</td><td>${r.rz.toFixed(1)}</td></tr>`).join("");
    function setFrame(i){ state.i=Math.max(0,Math.min(data.length-1,Number(i)||0)); update(); }
    slider.addEventListener("input", e=>setFrame(e.target.value));
    $("play").addEventListener("click",()=>{ state.playing=!state.playing; $("play").textContent=state.playing?"Ⅱ":"▶"; const tick=()=>{ if(!state.playing)return; setFrame((state.i+1)%data.length); state.timer=setTimeout(tick,1000/Math.max(source.fps||30,1));}; if(state.playing)tick(); else clearTimeout(state.timer);});
    $("reset").addEventListener("click",()=>{state.yaw=-.62; state.pitch=-.36; state.zoom=1; drawScene();});
    scene.addEventListener("pointerdown",e=>{state.drag=true; state.x=e.clientX; state.y=e.clientY; scene.setPointerCapture(e.pointerId);});
    scene.addEventListener("pointermove",e=>{if(!state.drag)return; state.yaw+=(e.clientX-state.x)*.008; state.pitch=Math.max(-1.35,Math.min(1.35,state.pitch+(e.clientY-state.y)*.008)); state.x=e.clientX; state.y=e.clientY; drawScene();});
    scene.addEventListener("pointerup",e=>{state.drag=false; scene.releasePointerCapture(e.pointerId);});
    scene.addEventListener("wheel",e=>{e.preventDefault(); state.zoom=Math.max(.42,Math.min(4,state.zoom*Math.exp(-e.deltaY*.0012))); drawScene();},{passive:false});
    window.addEventListener("resize", update);
    update();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a self-contained 6D pose trajectory HTML visualization.")
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--title", default="6D Pose Trajectory")
    parser.add_argument("--overlay-video", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = json.loads(args.pose_json.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = (
        HTML.replace("__DATA_JSON__", payload)
        .replace("__TITLE__", args.title)
        .replace("__OVERLAY_VIDEO__", args.overlay_video)
    )
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
