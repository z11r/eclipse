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
