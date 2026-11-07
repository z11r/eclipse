const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

// Execute the real frontend with a minimal DOM and controlled network timing.
function panel() {
  const elements = new Map();
  const element = () => ({disabled:false, classList:{toggle(){},add(){}},
    append(){},replaceChildren(){},setAttribute(){},addEventListener(){}});
  const calls = [];
  const state = {panic:false, simulated:false, output_state:'protected',
    output_connected:true, enabled:['email'], keywords:[], detector_health:{email:'ready'}};
  const context = vm.createContext({
    document:{getElementById(id){if(!elements.has(id))elements.set(id,element());return elements.get(id);},
      createElement:element},
    fetch(url){
      if(url === '/api/config') return new Promise(resolve => {calls.push(url);context.finishSave=()=>resolve({ok:true,json:async()=>({...state,panic:false})});});
      calls.push(url);
      if(url === '/api/panic') {state.panic=true;state.output_state='black';}
      return Promise.resolve({ok:true,json:async()=>({...state})});
    }
  });
  const source = fs.readFileSync(path.join(__dirname,'../eclipse/panel/static/app.js'),'utf8');
  vm.runInContext(source.slice(0,source.indexOf("$('categories-form').addEventListener")),context);
  context.initialState=state;
  vm.runInContext('renderState(initialState); configDirty=true;',context);
  return {context,calls,elements};
}

test('panic remains available during a delayed save and save cannot undo its display',async()=>{
  const {context,calls,elements}=panel();
  const saving=vm.runInContext("saveConfig('categories')",context);
  vm.runInContext('renderState(initialState)',context); // polling while saving
  assert.equal(elements.get('panic-button').disabled,false);
  await vm.runInContext("sendSafety('/api/panic')",context);
  assert.deepEqual(calls.slice(0,2),['/api/config','/api/panic']);
  assert.equal(elements.get('status-heading').textContent,'Emergency blackout');
  context.finishSave();
  await saving;
  assert.equal(elements.get('status-heading').textContent,'Emergency blackout');
  assert.equal(elements.get('resume-button').disabled,false);
});

test('duplicate safety requests are suppressed without blocking settings',async()=>{
  const {context,calls}=panel();
  const first=vm.runInContext("sendSafety('/api/panic')",context);
  const duplicate=vm.runInContext("sendSafety('/api/resume')",context);
  await Promise.all([first,duplicate]);
  assert.equal(calls.filter(url=>url==='/api/panic').length,1);
  assert.equal(calls.includes('/api/resume'),false);
});

test('pending settings stay visibly pending until a later acknowledgement',async()=>{
  const {context,elements}=panel();
  const saving=vm.runInContext("saveConfig('categories')",context);
  context.initialState.config_status='pending';
  context.finishSave();
  await saving;
  assert.match(elements.get('settings-status').textContent,/pending/);
  assert.equal(elements.get('save-settings').disabled,true);
  assert.equal(elements.get('panic-button').disabled,false);
  assert.equal(vm.runInContext('configDirty',context),true);
  context.initialState.config_status='applied';
  await vm.runInContext('refreshState()',context);
  assert.equal(elements.get('settings-status').textContent,'Settings applied');
  assert.equal(vm.runInContext('configDirty',context),false);
});

test('late failure displays the error and preserves edits for retry',async()=>{
  const {context,elements}=panel();
  const saving=vm.runInContext("saveConfig('categories')",context);
  context.initialState.config_status='pending';
  context.finishSave();
  await saving;
  context.initialState.config_status='failed';
  context.initialState.config_message='Settings were not applied. Previous settings remain active.';
  await vm.runInContext('refreshState()',context);
  assert.match(elements.get('settings-status').textContent,/not applied/);
  assert.equal(elements.get('save-settings').disabled,false);
  assert.equal(vm.runInContext('configDirty',context),true);
});

function logResponse(context, data) {
  context.fetch=async()=>({ok:true,json:async()=>data});
  return vm.runInContext('refreshEvents()',context);
}
const catchEvent=(id,seq,count=1)=>({incident_id:id,seq,count,time:seq,source:'screen',category:'email'});

test('log updates use stable incident IDs despite skipped polls and duplicate responses',async()=>{
  const {context}=panel();
  await logResponse(context,{session_id:'one',last:1,events:[catchEvent(1,1)]});
  const update={session_id:'one',last:8,events:[catchEvent(1,8,8),catchEvent(7,7)]};
  await logResponse(context,update);
  await logResponse(context,update);
  assert.equal(vm.runInContext('eventRows.length',context),2);
  assert.equal(vm.runInContext('eventRows.find(e=>e.incident_id===1).count',context),8);
});

test('backend restart clears stale rows and cursor even when new sequence is higher',async()=>{
  const {context}=panel();
  await logResponse(context,{session_id:'old',last:3,events:[catchEvent(1,3)]});
  let requested;
  context.fetch=async url=>{requested=url;return {ok:true,json:async()=>({session_id:'new',last:10,events:[catchEvent(2,10)]})};};
  await vm.runInContext('refreshEvents()',context);
  assert.match(requested,/session_id=old/);
  assert.equal(vm.runInContext('eventRows.length',context),1);
  assert.equal(vm.runInContext('eventRows[0].incident_id',context),2);
  await logResponse(context,{session_id:'empty',last:0,events:[]});
  assert.equal(vm.runInContext('eventRows.length',context),0);
  assert.equal(vm.runInContext('lastSeq',context),0);
  await logResponse(context,{session_id:'empty',last:1,events:[catchEvent(1,1)]});
  assert.equal(vm.runInContext('eventRows.length',context),1);
});

test('log failure retains history and later polling resumes with the same cursor',async()=>{
  const {context,elements}=panel();
  await logResponse(context,{session_id:'one',last:4,events:[catchEvent(1,4)]});
  context.fetch=async()=>{throw Error('offline');};
  await vm.runInContext('refreshEvents()',context);
  assert.equal(elements.get('log-connection').textContent,'Activity log unavailable');
  assert.equal(vm.runInContext('eventRows.length',context),1);
  assert.equal(vm.runInContext('lastSeq',context),4);
  await logResponse(context,{session_id:'one',last:5,events:[catchEvent(1,5,2)]});
  assert.equal(vm.runInContext('eventRows.length',context),1);
  assert.equal(elements.get('log-connection').textContent,'Log connected');
});
