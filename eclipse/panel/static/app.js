'use strict';

// The terminal launch link carries a token in its fragment, never in a query.
function loadPanelToken(){
  const fragment=new URLSearchParams(window.location.hash.slice(1));
  const launched=fragment.get('token');
  if(launched){
    window.history.replaceState(null,'',window.location.pathname+window.location.search);
    try{window.sessionStorage.setItem('eclipse-panel-token',launched);}catch(_){}
    return launched;
  }
  try{return window.sessionStorage.getItem('eclipse-panel-token') || '';}catch(_){return '';}
}
const panelToken=loadPanelToken();

const TEXT = [
  ['two_fa','2FA verification codes'], ['email','Email addresses'],
  ['phone','Phone numbers'], ['api_key','API keys & tokens'],
  ['card','Payment card candidates'], ['keyword','Custom keywords']
];
const VISUAL = [
  ['street_sign','Street signs'], ['house_number','House numbers'],
  ['licence_plate','Licence plates'], ['mail_label','Mail & package labels'],
  ['window_view','Views through windows'], ['map','Open maps'],
  ['storefront','Storefront names']
];
const CATEGORY_NAMES = Object.fromEntries([...TEXT,...VISUAL]);
const $ = (id) => document.getElementById(id);
let serverState = null;
let draftEnabled = new Set();
let draftKeywords = [];
let configDirty = false;
let keywordDirty = false;
let connected = false;
let lastSeq = 0;
let logSession = null;
let eventRows = [];
let logAvailable = true;
let statePollInFlight = false;
let logPollInFlight = false;
let configPending = false;
let detectorConfigPending = false;
let safetyPending = false;
let revealTimeouts = new Map();

async function api(path, options={}) {
  const response = await fetch(path, {
    ...options,
    credentials:'same-origin', cache:'no-store',
    headers:{'Accept':'application/json', ...(options.body?{'Content-Type':'application/json'}:{}), ...(options.headers||{}), 'Authorization':`Bearer ${panelToken}`}
  });
  if(response.status===401){
    throw Error('Panel authorization expired or missing. Open the current launch link from the Eclipse terminal.');
  }
  if (!response.ok) {
    let detail='Request failed ('+response.status+')';
    try { const data=await response.json(); if (typeof data.detail === 'string') detail=data.detail; } catch (_) {}
    throw Error(detail);
  }
  return response.json();
}
function notice(message, info=false) {
  const n=$('notice'); n.textContent=message;n.className='notice'+(info?' info':'');
}
function clearNotice() { $('notice').classList.add('hidden'); }
function setConnection(value) {
  connected=value;
  $('connection').textContent=value?'Connected to Eclipse':'Not connected to Eclipse';
  $('connection').className='connection '+(value?'good':'bad');
  if (!value) {
    $('status-heading').textContent='Status unavailable';
    $('status-description').textContent='Cannot verify protection state. Do not assume the stream is safe.';
    $('safety-label').textContent='UNVERIFIED';
    $('status-orb').className='status-orb black';
    $('panic-button').disabled=true;$('resume-button').disabled=true;
    $('save-settings').disabled=true;$('save-keywords').disabled=true;
    $('add-keyword').disabled=true;
    $('settings-status').textContent='Disconnected';
    $('metric-output').textContent='Unknown';
    $('metric-held').textContent='—';
    $('metric-detectors').textContent='—';
    $('detector-metric').style.setProperty('--ready','0');
    $('metric-detector-sub').textContent='Cannot verify detector health';
    $('output-chip').textContent='Output unverified';
    $('mode-chip').textContent='Mode unverified';
    $('irl-label').textContent='Unknown';
    const logState=$('log-connection');
    if(logState){logState.textContent='Service disconnected';logState.className='log-health bad';}

  }
}
function renderState(state) {
  if(detectorConfigPending && state.config_status==='applied'){
    configDirty=false;keywordDirty=false;
    notice('Settings applied and private keywords saved.',true);
  }
  serverState=state;setConnection(true);
  detectorConfigPending=state.config_status==='pending';
  const mock=Boolean(state.simulated);
  $('mock-warning').classList.toggle('hidden',!mock);
  $('mode-chip').textContent=state.mode==='irl'?'IRL mode':'Desktop mode';
  $('output-chip').textContent=state.output_connected?'Virtual camera connected':'Virtual camera disconnected';
  $('metric-output').textContent=state.output_connected?'Connected':'Offline';
  $('metric-held').textContent=String(state.held_frames ?? '—');
  const detectorStates=Object.values(state.detector_health || {});
  const ready=detectorStates.filter(s=>s==='ready').length;
  const degraded=detectorStates.filter(s=>s==='degraded').length;
  $('metric-detectors').textContent=`${ready}/${detectorStates.length} ready`;
  $('detector-metric').style.setProperty('--ready',String(detectorStates.length?ready/detectorStates.length:0));
  const unavailable=detectorStates.filter(s=>s==='unavailable').length;
  $('metric-detector-sub').textContent=[degraded?`${degraded} degraded`:null,unavailable?`${unavailable} unavailable`:null].filter(Boolean).join(' · ') || 'All detectors available';
  $('irl-label').textContent=state.irl_available?'Available':'Planned';
  let label,description, safety,orb='status-orb';
  if(mock){label='Demo / simulation only';description='The API is running, but no video is being captured or protected.';safety='NOT ACTIVE';}
  else if(state.panic && state.output_state==='black'){label='Emergency blackout';description='The pipeline reports black output. Resume is required to continue.';safety='BLACKOUT';orb+=' black';}
  else if(state.panic){label='Blackout requested';description='Black output is not yet confirmed. Verify OBS output.';safety='UNVERIFIED';orb+=' black';}
  else if(state.output_state==='protected' && state.output_connected && degraded===0){label='Protected output active';description='The pipeline reports protected frames reaching the output.';safety='PROTECTED';orb+=' protected';}
  else if(state.output_state==='covered'){label='Fallback cover active';description='Processing is degraded or delayed. The pipeline reports covered output.';safety='COVERED';orb+=' black';}
  else {label='Protection not verified';description=state.message || 'Capture or protected output is not available.';safety='UNVERIFIED';}
  $('status-heading').textContent=label;$('status-description').textContent=description;
  $('safety-label').textContent=safety;$('status-orb').className=orb;
  $('panic-button').disabled=safetyPending || !connected || Boolean(state.panic);
  $('resume-button').disabled=safetyPending || !connected || !state.panic;
  $('panic-hint').textContent=mock?'Mock controls only — no real OBS output exists.':state.panic?'Blackout stays active until explicit resume.':'Use panic to request immediate black output.';
  // Preserve any user edits while polling; an unchanged draft follows authoritative state.
  if(!configDirty) draftEnabled=new Set(state.enabled);
  if(!keywordDirty) draftKeywords=[...state.keywords];
  renderCategories(state);renderKeywords();
}
function renderCategories(state) {
  for(const [container,items] of [['text-categories',TEXT],['visual-categories',VISUAL]]) {
    const parent=$(container);
    parent.replaceChildren();
    for(const [id,label] of items) {
      const health=state.detector_health?.[id] || 'unavailable';
      const row=document.createElement('label');row.className='category-row';
      const info=document.createElement('div');
      const name=document.createElement('div');name.className='category-name';name.textContent=label;
      const sub=document.createElement('div');sub.className='category-sub';
      sub.textContent=health==='ready'?'Available':health==='degraded'?'Degraded — fallback may be active':'Detector unavailable';
      info.append(name,sub);
      const toggle=document.createElement('span');toggle.className='switch';
      const input=document.createElement('input');input.type='checkbox';input.checked=draftEnabled.has(id);
      input.disabled=!connected || configPending || detectorConfigPending;input.setAttribute('aria-label',`Enable ${label}`);
      input.addEventListener('change',()=>{
        if(input.checked)draftEnabled.add(id);else draftEnabled.delete(id);
        configDirty=true;updateSaveControls();
      });
      const track=document.createElement('span');track.className='switch-track';toggle.append(input,track);
      row.append(info,toggle);parent.append(row);
    }
  }
  $('category-count').textContent=`${draftEnabled.size} / ${TEXT.length+VISUAL.length} ENABLED`;
  updateSaveControls();
}
function renderKeywords(){
  const parent=$('keyword-list');parent.replaceChildren();
  if(draftKeywords.length===0){const empty=document.createElement('div');empty.className='empty';empty.textContent='No keywords configured';parent.append(empty);}
  for(const [index,keyword] of draftKeywords.entries()){
    const item=document.createElement('div');item.className='keyword-item';
    const val=document.createElement('span');val.className='keyword-value';val.textContent='•'.repeat(Math.min(keyword.length,18));
    const reveal=document.createElement('button');reveal.className='tiny-button';reveal.type='button';reveal.textContent='Hold to reveal';
    reveal.setAttribute('aria-label','Hold to reveal keyword');
    const show=()=>{val.textContent=keyword;val.classList.add('revealed');};
    const hide=()=>{val.textContent='•'.repeat(Math.min(keyword.length,18));val.classList.remove('revealed');};
    reveal.addEventListener('pointerdown',show);reveal.addEventListener('pointerup',hide);
    reveal.addEventListener('pointerleave',hide);reveal.addEventListener('pointercancel',hide);
    reveal.addEventListener('keydown',e=>{if(e.key===' '||e.key==='Enter')show();});
    reveal.addEventListener('keyup',hide);reveal.addEventListener('blur',hide);
    const remove=document.createElement('button');remove.className='tiny-button';remove.textContent='Remove';remove.type='button';
    remove.setAttribute('aria-label','Remove keyword');remove.disabled=!connected||configPending||detectorConfigPending;
    remove.addEventListener('click',()=>{draftKeywords.splice(index,1);keywordDirty=true;renderKeywords();updateSaveControls();});
    item.append(val,reveal,remove);parent.append(item);
  }
  updateSaveControls();
}
function updateSaveControls(){
  $('save-settings').disabled=!connected||configPending||detectorConfigPending||!configDirty;
  $('save-keywords').disabled=!connected||configPending||detectorConfigPending||!keywordDirty;
  $('add-keyword').disabled=!connected||configPending||detectorConfigPending;
  $('settings-status').textContent=configPending?'Submitting settings…':detectorConfigPending?'Settings pending — waiting for detector':serverState?.config_status==='failed'?(serverState.config_message || 'Settings failed — retry saving'):configDirty?'Unsaved category changes':'Settings applied';
  $('category-count').textContent=`${draftEnabled.size} / ${TEXT.length+VISUAL.length} ENABLED`;
}
async function saveConfig(kind){
  if(!serverState || configPending || detectorConfigPending)return;
  configPending=true;updateSaveControls();
  try{
    // Always send full lists, preserving the other unsaved section as draft.
    const state=await api('/api/config',{method:'PUT',body:JSON.stringify({enabled:[...draftEnabled],keywords:[...draftKeywords]})});
    if(state.config_status==='pending'){
      detectorConfigPending=true;
      notice('Settings pending. Waiting for the detector to confirm application.',true);
    } else {
      configDirty=false;keywordDirty=false;
      draftEnabled=new Set(state.enabled);draftKeywords=[...state.keywords];
      notice('Settings applied and private keywords saved.',true);
    }
    await refreshState();
  }catch(e){notice('Could not save configuration: '+e.message);await refreshState();}
  finally{configPending=false;updateSaveControls();}
}
async function sendSafety(route){
  if(safetyPending || !connected)return;
  safetyPending=true;$('panic-button').disabled=true;$('resume-button').disabled=true;
  try{
    const state=await api(route,{method:'POST'});clearNotice();renderState(state);
    if(state.simulated)notice('Mock action applied. No real stream or OBS output was changed.',true);
    else if(route.endsWith('panic') && state.output_state!=='black')notice('Blackout requested, but black output has not been confirmed.');
  }catch(e){notice('Unable to confirm pipeline action: '+e.message);await refreshState();}
  finally{safetyPending=false;if(serverState && connected)renderState(serverState);}
}
function renderEvents(){
  const parent=$('event-list');parent.replaceChildren();
  $('events-count').textContent=`${eventRows.length} EVENTS`;
  if(eventRows.length===0){const el=document.createElement('div');el.className='empty';el.textContent='No detection events reported.';parent.append(el);return;}
  for(const event of [...eventRows].sort((a,b)=>b.seq-a.seq).slice(0,60)){
    const row=document.createElement('div');row.className='event-row';
    const icon=document.createElement('span');icon.className='event-icon';icon.textContent='◇';icon.setAttribute('aria-hidden','true');
    const copy=document.createElement('div');copy.className='event-copy';
    const name=document.createElement('div');name.className='event-name';name.textContent=CATEGORY_NAMES[event.category] || 'Privacy detection';
    const meta=document.createElement('div');meta.className='event-meta';meta.textContent=`${event.source} · ${new Date(event.time*1000).toLocaleTimeString()}`;
    copy.append(name,meta);
    const count=document.createElement('span');count.className='event-count';count.textContent=`×${event.count}`;
    row.append(icon,copy,count);parent.append(row);
  }
}
async function refreshState(){
  if(statePollInFlight)return;
  statePollInFlight=true;
  try{const state=await api('/api/state');renderState(state);}
  catch(e){setConnection(false);notice(e.message || 'Cannot reach the local Eclipse server. Current protection status is unknown.');}
  finally{statePollInFlight=false;}
}
async function refreshEvents(){
  if(!connected||logPollInFlight)return;
  logPollInFlight=true;
  try {
    const sessionQuery=logSession===null?'':`&session_id=${encodeURIComponent(logSession)}`;
    const data=await api(`/api/log?after=${lastSeq}${sessionQuery}`);
    if(data.session_id!==logSession){
      eventRows=[];
      lastSeq=0;
      logSession=data.session_id;
    }
    for(const event of data.events){
      const oldIndex=eventRows.findIndex(old=>old.incident_id===event.incident_id);
      if(oldIndex<0)eventRows.push(event);
      else if(event.seq>eventRows[oldIndex].seq)eventRows[oldIndex]=event;
    }
    lastSeq=Math.max(lastSeq,data.last);
    eventRows.sort((a,b)=>b.seq-a.seq);
    if(eventRows.length>80)eventRows=eventRows.slice(0,80);
    logAvailable=true;
    const status=$('log-connection');
    if(status){status.textContent='Log connected';status.className='log-health good';}
    renderEvents();
  } catch(e) {
    logAvailable=false;
    const status=$('log-connection');
    if(status){status.textContent='Activity log unavailable';status.className='log-health bad';}
    notice('Activity log temporarily unavailable: '+e.message);
  } finally {logPollInFlight=false;}
}
$('categories-form').addEventListener('submit',e=>{e.preventDefault();saveConfig('categories');});
$('keyword-form').addEventListener('submit',e=>{
  e.preventDefault();
  const input=$('keyword-input'),keyword=input.value;
  if(!keyword.trim()||keyword.length>120||/[\x00-\x1f]/.test(keyword)){notice('Enter 1–120 printable characters.');return;}
  if(draftKeywords.some(k=>k.toLocaleLowerCase()===keyword.toLocaleLowerCase())){notice('That keyword is already in your list.');return;}
  if(draftKeywords.length>=100){notice('Maximum 100 keywords.');return;}
  draftKeywords.push(keyword);input.value='';keywordDirty=true;renderKeywords();clearNotice();
});
$('save-keywords').addEventListener('click',()=>saveConfig('keywords'));
$('panic-button').addEventListener('click',()=>sendSafety('/api/panic'));
$('resume-button').addEventListener('click',()=>sendSafety('/api/resume'));
refreshState().then(refreshEvents);
setInterval(refreshState,1000);
setInterval(refreshEvents,500);

// Lightweight anchor navigation; the panel stays a single-page control surface.
for (const link of document.querySelectorAll('.side-nav .nav-item')) {
  link.addEventListener('click', () => {
    for (const item of document.querySelectorAll('.side-nav .nav-item')) item.classList.remove('active');
    link.classList.add('active');
  });
}
