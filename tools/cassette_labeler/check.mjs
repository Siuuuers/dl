import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const original = '0 0.50000000 0.50000000 0.50000000 0.50000000 0.25000000 0.75000000 2 0.75000000 0.75000000 2';
const edited = original.replace('0.50000000', '0.55000000');

function fixture(splits = ['train', 'val'], initial = original) {
  const saved = new Map();
  const state = {writes: 0, picks: 0, prompts: 0, discard: false, fail: '', closeGate: null};
  const missing = () => Object.assign(new Error('Not found'), {name: 'NotFoundError'});
  const images = new Map(), labels = new Map();
  for (const split of splits) {
    if (initial !== null) saved.set(`${split}/a.txt`, initial + (initial ? '\n' : ''));
    saved.set(`${split}/b.txt`, '');
    images.set(split, {
      kind: 'directory',
      async *entries() {
        for (const name of ['a.png', 'b.png']) yield [name, {kind: 'file', async getFile() { return {name}; }}];
      },
    });
    labels.set(split, {
      async getFileHandle(name, {create = false} = {}) {
        const path = `${split}/${name}`;
        if (!saved.has(path) && !create) throw missing();
        return {
          async getFile() { return {async text() { return saved.get(path); }}; },
          async createWritable() {
            let staged;
            return {
              async write(text) {
                state.writes++;
                if (state.fail === 'write') throw Error('Write failed');
                staged = text;
              },
              async close() {
                if (state.closeGate) await state.closeGate;
                if (state.fail === 'close') throw Error('Close failed');
                saved.set(path, staged);
              },
            };
          },
        };
      },
    });
  }
  const root = {
    async getDirectoryHandle(name) {
      const children = name === 'images' ? images : labels;
      return {
        async *entries() { yield* children; },
        async getDirectoryHandle(split) {
          if (!children.has(split)) throw missing();
          return children.get(split);
        },
      };
    },
  };
  const elements = Object.fromEntries([...html.matchAll(/<\w+\b([^>]*\bid="([^"]+)"[^>]*)>/g)].map(([, attributes, id]) => [id, {
    value: '', textContent: '', disabled: /\bdisabled\b/.test(attributes), children: [],
    replaceChildren(...children) { this.children = children; this.value = children[0]?.value ?? ''; },
  }]));
  const document = {getElementById: id => elements[id], activeElement: null};
  Object.assign(elements.canvas, {
    width: 1280, height: 720,
    getContext: () => new Proxy({}, {get: () => () => {}, set: () => true}),
    getBoundingClientRect: () => ({left: 20, top: 10, width: 640, height: 360}),
    setPointerCapture() {},
  });
  const showDirectoryPicker = async () => { state.picks++; return root; };
  const window = {showDirectoryPicker};
  vm.runInNewContext(script, {
    document, window, showDirectoryPicker,
    confirm: () => { state.prompts++; return state.discard; },
    Option: function (text, value) { this.text = text; this.value = String(value); },
    createImageBitmap: async () => ({width: 1280, height: 720, close() {}}),
  });
  async function event(id, type) {
    const element = elements[id];
    assert.ok(element, `Missing control: ${id}`);
    document.activeElement = element;
    if (!element.disabled && (id === 'open' || !elements.controls.disabled)) await element[type]?.({target: element});
    await new Promise(setImmediate);
  }
  return {
    elements, saved, state,
    click: id => event(id, 'onclick'),
    async select(id, value) { elements[id].value = String(value); await event(id, 'onchange'); },
    text(value) { elements.label.value = value; document.activeElement = elements.label; elements.label.oninput?.(); },
    unload() {
      const event = {prevented: false, preventDefault() { this.prevented = true; }};
      window.onbeforeunload(event);
      return event.prevented;
    },
    drag(mode, start, end = start) {
      elements.mode.value = mode; document.activeElement = elements.canvas;
      const pointer = p => ({button: 0, pointerId: 1, clientX: 20 + p[0] * 640, clientY: 10 + p[1] * 360});
      elements.canvas.onpointerdown(pointer(start));
      if (start !== end) {
        elements.canvas.onpointermove(pointer(start.map((v, i) => (v + end[i]) / 2)));
        elements.canvas.onpointermove(pointer(end));
      }
      elements.canvas.onpointerup(pointer(end));
    },
  };
}

async function open(splits, initial) {
  const app = fixture(splits, initial);
  await app.click('open');
  assert.ok(app.elements.files.children.length, app.elements.status.textContent);
  assert.equal(app.elements.controls.disabled, false);
  return app;
}

const checks = [
  ['Apply, Save, Undo, and empty labels round-trip', async () => {
    const app = await open();
    assert.equal(app.elements.label.value, original);
    app.text(edited); await app.click('apply');
    assert.equal(app.elements.label.value, edited);
    assert.equal(app.unload(), true);
    await app.click('save');
    assert.equal(String(app.elements.files.value), '0');
    assert.equal(app.saved.get('train/a.txt'), edited + '\n');
    assert.equal(app.unload(), false);
    await app.click('undo');
    assert.equal(app.elements.label.value, original);
    assert.equal(app.unload(), true);
    await app.click('save');
    assert.equal(app.saved.get('train/a.txt'), original + '\n');
    assert.equal(app.unload(), false);
    app.text(''); await app.click('apply'); await app.click('save');
    assert.equal(app.elements.objects.children.length, 0);
    assert.equal(app.saved.get('train/a.txt'), '');
  }],
  ['Draft survives selection and protects navigation and unload', async () => {
    const initial = original + '\n' + edited;
    const app = await open(undefined, initial);
    const draft = edited + '\n' + edited;
    app.text(draft);
    await app.select('objects', 1);
    assert.equal(app.elements.label.value, draft);
    assert.equal(app.unload(), true);
    await app.select('files', 1);
    assert.equal(app.state.prompts, 1);
    assert.equal(String(app.elements.files.value), '0');
    await app.click('open');
    assert.equal(app.state.prompts, 2);
    assert.equal(app.state.picks, 1);
    assert.equal(app.elements.label.value, draft);
    await app.click('reset');
    assert.equal(app.elements.label.value, initial);
    assert.equal(app.unload(), false);
    app.text(draft); app.state.discard = true;
    await app.select('files', 1);
    assert.equal(app.elements.label.value, '');
    assert.equal(app.unload(), false);
    assert.equal(app.elements.undo.disabled, true);
  }],
  ['Pending text blocks Save and mutations; Reset preserves the applied edit', async () => {
    const app = await open();
    app.text(edited); await app.click('apply');
    const draft = edited.replace('0.55000000', '0.60000000');
    app.text(draft);
    for (const id of ['save', 'add', 'remove', 'missingLeft', 'missingRight', 'undo']) {
      await app.click(id);
      assert.equal(app.elements.label.value, draft, id);
      assert.equal(app.elements.objects.children.length, 1, id);
    }
    app.drag('box', [0.1, 0.1], [0.2, 0.2]);
    assert.equal(app.state.writes, 0);
    assert.equal(app.saved.get('train/a.txt'), original + '\n');
    await app.click('reset');
    assert.equal(app.elements.label.value, edited);
    assert.equal(app.unload(), true);
    await app.click('undo');
    assert.equal(app.elements.label.value, original);
    assert.equal(app.unload(), false);
  }],
  ['Invalid Apply preserves the draft, rows, and Undo', async () => {
    const app = await open();
    app.text(edited); await app.click('apply');
    app.text('0 0.5'); await app.click('apply');
    assert.equal(app.elements.label.value, '0 0.5');
    assert.match(app.elements.status.textContent, /invalid/i);
    assert.equal(app.elements.objects.children.length, 1);
    await app.click('reset');
    assert.equal(app.elements.label.value, edited);
    await app.click('undo');
    assert.equal(app.elements.label.value, original);
  }],
  ['Unchanged Apply and point clicks preserve Undo', async () => {
    const app = await open();
    app.text(edited); await app.click('apply');
    app.text(edited + '\n'); await app.click('apply');
    app.drag('edit', [0.25, 0.75]);
    assert.equal(app.elements.label.value, edited);
    await app.click('undo');
    assert.equal(app.elements.label.value, original);
  }],
  ['Already missing points preserve Undo', async () => {
    for (const [id, point] of [['missingLeft', '0.25000000 0.75000000 2'], ['missingRight', '0.75000000 0.75000000 2']]) {
      const initial = original.replace(point, '0.00000000 0.00000000 0');
      const app = await open(undefined, initial);
      app.text(initial.replace('0.50000000', '0.55000000')); await app.click('apply');
      await app.click(id); await app.click('undo');
      assert.equal(app.elements.label.value, initial, id);
    }
  }],
  ['One-step Undo covers add, delete, missing points, boxes, and point edits', async () => {
    const edits = [
      ...['add', 'remove', 'missingLeft', 'missingRight'].map(id => [id, app => app.click(id)]),
      ['move box', app => app.drag('edit', [0.5, 0.5], [0.6, 0.55])],
      ['replace box', app => app.drag('box', [0.1, 0.1], [0.4, 0.4])],
      ['move point', app => app.drag('edit', [0.25, 0.75], [0.3, 0.65])],
      ['place left', app => app.drag('left', [0.2, 0.7])],
      ['place right', app => app.drag('right', [0.8, 0.7])],
    ];
    for (const [name, edit] of edits) {
      const app = await open();
      await edit(app);
      assert.notEqual(app.elements.label.value, original, name);
      await app.click('undo');
      assert.equal(app.elements.label.value, original, name);
      assert.equal(app.elements.undo.disabled, true, name);
      assert.equal(app.unload(), false, name);
    }
  }],
  ['Missing and empty label files can be labeled and saved', async () => {
    for (const initial of [null, '']) {
      const app = await open(undefined, initial);
      assert.equal(app.elements.label.value, '');
      app.text(original); await app.click('apply'); await app.click('save');
      assert.equal(app.saved.get('train/a.txt'), original + '\n');
    }
  }],
  ['A val-only draft folder opens and saves its matching label', async () => {
    const app = await open(['val']);
    assert.equal(app.elements.files.children[0].text, 'val/a.png');
    assert.equal(app.elements.label.value, original);
    app.text(edited); await app.click('apply'); await app.click('save');
    assert.equal(app.saved.get('val/a.txt'), edited + '\n');
  }],
  ['Write and close failures leave applied edits unsaved', async () => {
    for (const failure of ['write', 'close']) {
      const app = await open();
      app.text(edited); await app.click('apply'); app.state.fail = failure;
      await app.click('save');
      assert.match(app.elements.status.textContent, /failed/i);
      assert.equal(app.elements.label.value, edited);
      assert.equal(app.unload(), true);
      assert.equal(app.elements.undo.disabled, false);
      app.state.fail = ''; await app.click('save');
      assert.equal(app.saved.get('train/a.txt'), edited + '\n');
      assert.equal(app.unload(), false);
    }
  }],
  ['Save & Next waits for close before opening the next image', async () => {
    const app = await open(['val']);
    app.text(edited); await app.click('apply');
    let finishClose;
    app.state.closeGate = new Promise(resolve => { finishClose = resolve; });
    const saving = app.click('saveNext');
    await new Promise(setImmediate);
    assert.equal(app.state.writes, 1);
    assert.equal(app.elements.controls.disabled, true);
    assert.equal(String(app.elements.files.value), '0');
    assert.equal(app.elements.label.value, edited);
    assert.equal(app.saved.get('val/a.txt'), original + '\n');
    finishClose(); await saving;
    assert.equal(app.saved.get('val/a.txt'), edited + '\n');
    assert.equal(String(app.elements.files.value), '1');
    assert.equal(app.elements.label.value, '');
    assert.equal(app.elements.undo.disabled, true);
    assert.equal(app.unload(), false);
    assert.equal(app.state.prompts, 0);
  }],
  ['Save & Next saves and stays on the last image', async () => {
    const app = await open(['val']);
    await app.select('files', 1);
    app.text(edited); await app.click('apply'); await app.click('saveNext');
    assert.equal(app.saved.get('val/b.txt'), edited + '\n');
    assert.equal(String(app.elements.files.value), '1');
    assert.equal(app.elements.label.value, edited);
    assert.equal(app.elements.undo.disabled, false);
    assert.equal(app.unload(), false);
  }],
  ['Blocked Save & Next preserves the current image and permits retry', async () => {
    for (const failure of ['draft', 'box', 'write', 'close']) {
      const app = await open(['val']);
      app.text(edited);
      if (failure !== 'draft') await app.click('apply');
      if (failure === 'box') await app.click('add');
      if (failure === 'write' || failure === 'close') app.state.fail = failure;
      const pending = app.elements.label.value;
      await app.click('saveNext');
      assert.equal(String(app.elements.files.value), '0', failure);
      assert.equal(app.elements.label.value, pending, failure);
      assert.equal(app.saved.get('val/a.txt'), original + '\n', failure);
      assert.equal(app.unload(), true, failure);
      assert.equal(app.elements.controls.disabled, false, failure);
      if (failure === 'draft') await app.click('apply');
      else if (failure === 'box') await app.click('undo');
      else app.state.fail = '';
      await app.click('saveNext');
      assert.equal(app.saved.get('val/a.txt'), edited + '\n', failure);
      assert.equal(String(app.elements.files.value), '1', failure);
      assert.equal(app.elements.label.value, '', failure);
      assert.equal(app.unload(), false, failure);
    }
  }],
];

let failures = 0;
for (const [name, check] of checks) {
  try { await check(); console.log(`PASS ${name}`); }
  catch (error) { failures++; console.error(`FAIL ${name}\n${error.message}`); }
}
console.log(`${checks.length - failures}/${checks.length} focused labeler checks passed (DOM and file handles stubbed).`);
process.exitCode = failures ? 1 : 0;
