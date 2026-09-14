from __future__ import annotations

import json

HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Activation Oracle · Training comparison</title>
<style>
:root{color-scheme:dark;--bg:#111113;--fg:#ececec;--muted:#a4a4ad;--line:#303035;--panel:#1a1a1d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;letter-spacing:0}
main,header,nav{max-width:1400px;margin:auto;padding-left:28px;padding-right:28px}
header{padding-top:30px;padding-bottom:18px}
h1{font-size:26px;line-height:1.25;margin:5px 0 10px;font-weight:600}
h2{font-size:20px;margin:0 0 8px;font-weight:600}
h3{font-size:15px;margin:0 0 8px;font-weight:600}
p{margin:0 0 12px}.muted,.caption{color:var(--muted)}.caption{font-size:12px}
.eyebrow{text-transform:uppercase;font-size:11px;letter-spacing:0;color:var(--muted)}
nav{display:flex;gap:24px;flex-wrap:wrap;padding-top:12px;padding-bottom:12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
a{color:#9aceff;text-decoration:none}a:hover{text-decoration:underline}
section{padding:30px 0;border-bottom:1px solid var(--line);scroll-margin-top:16px}
.legend{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:14px}
.legend label{display:inline-flex;align-items:center;gap:7px;cursor:pointer;user-select:none}
.legend input{margin:0}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;flex:none}
.table-wrap{overflow-x:auto;width:100%;margin:14px 0 24px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:700px}
td,th{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
td:first-child,th:first-child{text-align:left}
th{font-size:12px;color:var(--muted);font-weight:500}
tbody tr:hover{background:#202024}
.best-value{font-weight:700;background:#24262b}
.group td{font-size:12px;color:var(--muted);padding-top:18px;background:var(--panel)}
.step{display:block;font-size:11px;color:var(--muted)}
.plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:26px 24px;margin-top:20px}
figure{margin:0;min-width:0}
svg{display:block;width:100%;height:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:visible}
svg text{font-family:ui-sans-serif,system-ui,sans-serif}
.description{padding:18px 0;border-top:1px solid var(--line)}
.description h3{display:flex;align-items:center;gap:8px}
.description p{max-width:1000px}
code{font-size:12px;overflow-wrap:anywhere}
.note{padding:12px 0 12px 14px;border-left:3px solid #f1bd55;color:var(--muted);margin:16px 0}
footer{padding:22px 0 35px;color:var(--muted);font-size:12px}
.tooltip{position:fixed;z-index:10;display:none;pointer-events:none;background:#ececec;color:#111113;padding:8px 10px;border-radius:4px;max-width:290px;font-size:12px;white-space:pre-line;box-shadow:0 3px 15px #0008}
@media(max-width:800px){main,header,nav{padding-left:16px;padding-right:16px}.plots{grid-template-columns:1fr}h1{font-size:22px}nav{gap:10px 18px}section{padding:24px 0}}
</style>
</head>
<body>
<header>
<div class="eyebrow">Qwen3-VL-4B-Instruct · Activation Oracle</div>
<h1>Training comparison</h1>
<p class="muted">Final checkpoints, best evaluated checkpoints, and validation through training.</p>
<div id="legend" class="legend"></div>
</header>
<nav aria-label="Report sections"><a href="#final">Final checkpoints</a><a href="#best">Best checkpoints</a><a href="#evolution">Training evolution</a><a href="#experiments">Experiments</a></nav>
<main>
<section id="final">
<h2>Final checkpoint evaluation</h2>
<p class="muted">Accuracy is shown in percent; n is the number of evaluated items.</p>
<p class="note">Activation difference uses the September 8 closed-set reevaluation at its final step. Earlier points retain the September 4 protocol. Target-family comparisons across those protocols are not strictly like-for-like.</p>
<div id="final-table"></div>
</section>
<section id="best">
<h2>Best evaluated checkpoints</h2>
<p class="muted">One step per experiment, selected by the unweighted mean answer accuracy across VSR, GQA, COCO presence, SNLI-VE, and caption prediction. All bars and radar vertices use that same step; ties select the earliest step.</p>
<div id="best-table"></div>
<div class="plots" id="best-plots"></div>
<p class="caption">Plot axes follow the selected experiments. Evaluated steps do not imply that adapter files were saved at every step. Binary chance is 50%; caption exact-match and target-family tasks have different baselines.</p>
</section>
<section id="evolution">
<h2>Validation through training</h2>
<p class="muted">Answer accuracy versus optimizer step. The outlined square marks the corrected Activation difference endpoint; preceding target values use the earlier protocol.</p>
<div class="plots" id="evolution-plots"></div>
<p class="caption">Plot axes follow the selected experiments.</p>
</section>
<section id="experiments">
<h2>Experiment setups</h2>
<p class="muted">Shared recipe: Qwen3-VL-4B-Instruct; frozen vision tower; text-side LoRA r=64, alpha=128, dropout=0.05. Source layers 9 / 18 / 27; decoder injection layer 1 with coefficient 1.0. Learning rate 1e-5; one epoch; batch 4 per rank; gradient accumulation 1; gradient checkpointing; seed 42.</p>
<p class="muted">Training caps: visual SPQA 150,000; VSR, GQA and COCO presence 6,000 each; caption prediction 100,000 in each of the single- and multi-activation modes. All standard and four target validation families enabled; evaluation at a regular step interval plus the final step.</p>
<div id="descriptions"></div>
</section>
<footer>Source: saved training evaluation histories; corrected final Activation difference evaluation from 20260908_adiff_final_closedset. All percentages use the stored metrics, without rounding before aggregation. No new training or evaluation was run.</footer>
</main>
<div id="tooltip" class="tooltip" role="status"></div>
<script id="report-data" type="application/json">"""

TAIL = r"""</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("report-data").textContent);
const runs=data.runs;
const standard=["classification_vsr","classification_gqa_yesno","classification_coco_presence","classification_snli_ve","coco_captions_past_lens"];
const targets=["visual_taboo","visual_user_attribute","visual_ssc","visual_personaqa"];
const datasets=[...standard,...targets];
const labels={classification_vsr:"VSR",classification_gqa_yesno:"GQA yes/no",classification_coco_presence:"COCO presence",classification_snli_ve:"SNLI-VE",coco_captions_past_lens:"Caption prediction",visual_taboo:"Visual Taboo",visual_user_attribute:"User attribute",visual_ssc:"SSC",visual_personaqa:"PersonaQA",standard_mean:"Standard mean",target_mean:"Target mean"};
const esc=s=>String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
const pct=v=>v==null?"N/A":(v*100).toFixed(2)+"%";
const metric=(p,d)=>p.metrics["eval_ans_correct/"+d];
const average=(p,ds)=>ds.reduce((s,d)=>s+metric(p,d),0)/ds.length;
const value=(p,d)=>d==="standard_mean"?average(p,standard):d==="target_mean"?average(p,targets):metric(p,d);
const best=r=>r.points.reduce((a,p)=>average(p,standard)>average(a,standard)?p:a,r.points[0]);
const swatch=r=>'<i class="swatch" style="background:'+r.color+'"></i>';
function visibleRuns(){
 return runs.filter((_,i)=>document.getElementById("run-"+i).checked);
}
function axisRange(values){
 if(!values.length) throw new Error("axisRange requires at least one value");
 let lo=Math.min(...values),hi=Math.max(...values);
 if(lo===hi){
  const pad=0.02;
  return {lo:lo-pad,hi:hi+pad};
 }
 const pad=(hi-lo)*0.08;
 return {lo:lo-pad,hi:hi+pad};
}
function yScale(v,lo,hi,top,height){
 return top+height*(1-(v-lo)/(hi-lo));
}
function tickLabel(v){
 const p=v*100;
 return (Math.abs(p-Math.round(p))<0.05?Math.round(p):p.toFixed(1))+"%";
}
const text=(x,y,t,extra="")=>'<text x="'+x+'" y="'+y+'" fill="#a4a4ad" font-size="11" '+extra+'>'+esc(t)+'</text>';
const tip=(s)=>' data-tip="'+esc(s)+'"';
function frame(title,body,w,h){return '<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(title)+'"><title>'+esc(title)+'</title>'+body+'</svg>';}
function figure(title,chart){return '<figure><h3>'+esc(title)+'</h3>'+chart+'</figure>';}
function grid(left,top,width,height,lo,hi){
 let s="";
 for(let i=0;i<5;i++){
  const t=i/4,v=lo+(hi-lo)*t,y=yScale(v,lo,hi,top,height);
  s+='<line x1="'+left+'" y1="'+y+'" x2="'+(left+width)+'" y2="'+y+'" stroke="#343438"/>'+text(left-9,y+4,tickLabel(v),'text-anchor="end"');
 }
 return s;
}
function emptyNote(host){
 host.innerHTML='<p class="muted">Select at least one experiment.</p>';
}
function renderAll(){
 const vis=visibleRuns();
 const n=vis.length;
 if(!n){
  emptyNote(document.getElementById("final-table"));
  emptyNote(document.getElementById("best-table"));
  emptyNote(document.getElementById("best-plots"));
  emptyNote(document.getElementById("evolution-plots"));
  return;
 }
 const selected=vis.map(best);
 const final=vis.map(r=>r.points.at(-1));
 const cols=vis.map(r=>'<th scope="col" style="color:'+r.color+'">'+esc(r.name)+'</th>').join("");
 const span=String(1+n);
 function cells(vs,counts){
  const present=vs.filter(v=>v!=null),max=present.length?Math.max(...present):null;
  return vs.map((v,i)=>'<td'+(v!=null&&v===max?' class="best-value"':'')+'>'+pct(v)+(counts&&v!=null?'<span class="step">n='+counts[i]+'</span>':"")+'</td>').join("");
 }
 let rows="";
 for(const [title,prefix] of [["Answer accuracy","eval_ans_correct/"],["Format accuracy","eval_format_correct/"]]){
  rows+='<tr class="group"><td colspan="'+span+'">'+title+'</td></tr>';
  if(prefix==="eval_ans_correct/")for(const d of ["standard_mean","target_mean"])rows+='<tr><td>'+labels[d]+'</td>'+cells(final.map(p=>value(p,d)))+'</tr>';
  for(const d of datasets)rows+='<tr><td>'+labels[d]+'</td>'+cells(final.map(p=>p.metrics[prefix+d]),vis.map(r=>r.counts[d]))+'</tr>';
 }
 rows+='<tr class="group"><td colspan="'+span+'">Target OOD answer accuracy</td></tr>';
 const ood=[...new Set(final.flatMap(p=>Object.keys(p.metrics).filter(k=>k.startsWith("eval_target_ood/"))))].sort();
 for(const k of ood){const parts=k.split("/");rows+='<tr><td>'+labels[parts[1]]+' / '+esc(parts[2].replaceAll("_"," "))+'</td>'+cells(final.map(p=>p.metrics[k]))+'</tr>';}
 document.getElementById("final-table").innerHTML='<div class="table-wrap"><table aria-label="Final checkpoint comparison"><thead><tr><th scope="col">Metric</th>'+cols+'</tr></thead><tbody>'+rows+'</tbody></table></div>';
 document.getElementById("best-table").innerHTML='<div class="table-wrap"><table aria-label="Selected best evaluated checkpoints"><thead><tr><th>Experiment</th><th>Selected step</th><th>Standard mean</th><th>Target mean</th></tr></thead><tbody>'+vis.map((r,i)=>'<tr><td style="color:'+r.color+'">'+esc(r.name)+'</td><td>'+selected[i].step.toLocaleString("en-US")+'</td><td>'+pct(average(selected[i],standard))+'</td><td>'+pct(average(selected[i],targets))+'</td></tr>').join("")+'</tbody></table></div>';
 function bars(ds,title){
  const values=ds.flatMap((d,j)=>vis.map((_,i)=>metric(selected[i],d)));
  const {lo,hi}=axisRange(values);
  const w=640,h=360,left=52,top=24,width=570,height=238;let s=grid(left,top,width,height,lo,hi);
  const group=width/ds.length,bw=Math.min(20,(group-20)/n);
  ds.forEach((d,j)=>{
   vis.forEach((r,i)=>{
    const v=metric(selected[i],d),x=left+group*j+(group-n*bw)/2+i*bw,y=yScale(v,lo,hi,top,height),barH=Math.max(1,top+height-y);
    s+='<rect x="'+x+'" y="'+y+'" width="'+(bw-2)+'" height="'+barH+'" fill="'+r.color+'"'+tip(r.name+' · '+labels[d]+'\n'+pct(v)+' · step '+selected[i].step)+'><title>'+esc(r.name+': '+pct(v))+'</title></rect>';
   });
   const lines=labels[d].split(" ");s+=text(left+group*(j+.5),286,lines.slice(0,2).join(" "),'text-anchor="middle"');if(lines.length>2)s+=text(left+group*(j+.5),301,lines.slice(2).join(" "),'text-anchor="middle"');
  });
  return frame(title,s,w,h);
 }
 function radar(ds,title){
  const values=ds.flatMap(d=>vis.map((_,i)=>metric(selected[i],d)));
  const {lo,hi}=axisRange(values);
  const w=640,h=390,cx=320,cy=198,radius=130;
  const frac=v=>(v-lo)/(hi-lo);
  const xy=(i,f)=>{const a=-Math.PI/2+i*2*Math.PI/ds.length;return [cx+Math.cos(a)*radius*f,cy+Math.sin(a)*radius*f];};
  let s="";
  for(const t of [0,1/3,2/3,1]){
   const v=lo+(hi-lo)*t;
   s+='<polygon points="'+ds.map((_,i)=>xy(i,t).join(",")).join(" ")+'" fill="none" stroke="#343438"/>'+text(cx+5,cy-radius*t+12,tickLabel(v));
  }
  ds.forEach((d,i)=>{const p=xy(i,1),q=xy(i,1.22);s+='<line x1="'+cx+'" y1="'+cy+'" x2="'+p[0]+'" y2="'+p[1]+'" stroke="#343438"/>'+text(q[0],q[1]+4,labels[d],'text-anchor="middle"');});
  vis.forEach((r,j)=>{
   s+='<polygon points="'+ds.map((d,i)=>xy(i,frac(metric(selected[j],d))).join(",")).join(" ")+'" fill="'+r.color+'" fill-opacity=".04" stroke="'+r.color+'" stroke-width="2"/>';
   ds.forEach((d,i)=>{const p=xy(i,frac(metric(selected[j],d)));s+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="4" fill="'+r.color+'"'+tip(r.name+' · '+labels[d]+'\n'+pct(metric(selected[j],d))+' · step '+selected[j].step)+'/>';});
  });
  return frame(title,s,w,h);
 }
 document.getElementById("best-plots").innerHTML=figure("Standard benchmarks · grouped bars",bars(standard,"Best checkpoint standard accuracy"))+figure("Target families · grouped bars",bars(targets,"Best checkpoint target accuracy"))+figure("Standard benchmarks · radar",radar(standard,"Best checkpoint standard radar"))+figure("Target families · radar",radar(targets,"Best checkpoint target radar"));
 function evolution(d){
  const values=vis.flatMap(r=>r.points.map(p=>value(p,d)));
  const {lo,hi}=axisRange(values);
  const maxStep=Math.max(...vis.flatMap(r=>r.points.map(p=>p.step)));
  const w=640,h=295,left=52,top=24,width=562,height=210;let s=grid(left,top,width,height,lo,hi);
  const ticks=[0];
  for(let x=5000;x<maxStep;x+=5000)ticks.push(x);
  if(ticks[ticks.length-1]!==maxStep)ticks.push(maxStep);
  for(const x of ticks)s+=text(left+width*x/maxStep,256,x===maxStep?x.toLocaleString("en-US"):x===0?"0":x/1000+"k",'text-anchor="middle"');
  s+=text(left+width/2,280,"Optimizer step",'text-anchor="middle"');
  vis.forEach(r=>{
   const pts=r.points.map(p=>[left+width*p.step/maxStep,yScale(value(p,d),lo,hi,top,height)]);
   s+='<polyline points="'+pts.map(p=>p.join(",")).join(" ")+'" fill="none" stroke="'+r.color+'" stroke-width="2.2"/>';
   pts.forEach((p,j)=>{
    const label=r.name+' · '+labels[d]+'\n'+pct(value(r.points[j],d))+' · step '+r.points[j].step+(r.overlay&&j===pts.length-1?'\nSeptember 8 closed-set reevaluation':"");
    if(r.overlay&&j===pts.length-1)s+='<rect x="'+(p[0]-5)+'" y="'+(p[1]-5)+'" width="10" height="10" fill="#1a1a1d" stroke="'+r.color+'" stroke-width="2"'+tip(label)+'/>';
    else s+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="3" fill="'+r.color+'"'+tip(label)+'/>';
   });
  });
  return frame(labels[d]+" evaluation history",s,w,h);
 }
 document.getElementById("evolution-plots").innerHTML=[...datasets,"standard_mean","target_mean"].map(d=>figure(labels[d],evolution(d))).join("");
}
document.getElementById("legend").innerHTML=runs.map((r,i)=>'<label>'+swatch(r)+'<input type="checkbox" id="run-'+i+'" checked>'+esc(r.name)+'</label>').join("");
document.getElementById("legend").addEventListener("change",renderAll);
document.getElementById("descriptions").innerHTML=runs.map(r=>'<article class="description"><h3>'+swatch(r)+esc(r.name)+'</h3><p>'+esc(r.description)+'</p><p class="caption"><code>'+esc(r.directory)+'</code></p><a href="../'+encodeURIComponent(r.directory)+'/results.html">Training report</a> · <a href="../'+encodeURIComponent(r.directory)+'/results.json">Recorded training metrics</a></article>').join("");
renderAll();
const tooltip=document.getElementById("tooltip");
document.addEventListener("pointermove",event=>{const target=event.target.closest("[data-tip]");if(!target){tooltip.style.display="none";return;}tooltip.textContent=target.dataset.tip;tooltip.style.display="block";tooltip.style.left=Math.max(8,Math.min(event.clientX+12,window.innerWidth-tooltip.offsetWidth-8))+"px";tooltip.style.top=Math.max(8,Math.min(event.clientY+12,window.innerHeight-tooltip.offsetHeight-8))+"px";});
document.addEventListener("pointerleave",()=>{tooltip.style.display="none";});
</script>
</body>
</html>
"""


def render_report(payload: dict) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    payload_json = payload_json.replace("</", "<\\/")
    return HEAD + payload_json + TAIL
