"use strict";
(() => {
  const viewport = document.getElementById("viewport");
  const tree = document.getElementById("tree");
  const namespace = "http://www.w3.org/2000/svg";
  let args = null, identity = null, counts = {}, inputs = new Map();
  let lastSent = null, disabled = false, lastHeight = 0;

  function send(type, payload = {}) {
    window.parent.postMessage({isStreamlitMessage: true, type, ...payload}, "*");
  }

  function commit(span, input) {
    const value = input.valueAsNumber;
    if (!Number.isSafeInteger(value) || value < 0 || !input.validity.valid) return false;
    if (counts[span] === value) return true;
    counts[span] = value;
    lastSent = crypto.randomUUID();
    send("streamlit:setComponentValue", {
      dataType: "json", value: {tree_id: identity, event_id: lastSent, rounds: {...counts}}
    });
    return true;
  }

  function rebuild() {
    tree.replaceChildren();
    inputs = new Map();
    const svg = document.createElementNS(namespace, "svg");
    svg.setAttribute("aria-hidden", "true");
    tree.append(svg);
    for (const edge of args.edges) {
      const line = document.createElementNS(namespace, "line");
      line.dataset.parent = edge.parent;
      line.dataset.child = edge.child;
      line.setAttribute("stroke", "#94a3b8");
      line.setAttribute("stroke-width", "1.5");
      svg.append(line);
    }
    for (const node of args.nodes) {
      const input = document.createElement(args.editable ? "input" : "span");
      input.className = "node";
      input.dataset.span = node.span;
      input.title = `${node.label} — distillation rounds`;
      input.setAttribute("aria-label", `${node.label}: distillation rounds`);
      if (args.editable) {
        input.type = "number";
        input.min = "0";
        input.step = "1";
        input.inputMode = "numeric";
        input.value = String(counts[node.span]);
        input.addEventListener("input", () => commit(node.span, input));
        input.addEventListener("change", () => {
          if (!commit(node.span, input)) input.reportValidity();
        });
        input.addEventListener("blur", () => {
          if (!commit(node.span, input)) input.value = String(counts[node.span]);
        });
        input.addEventListener("keydown", event => {
          if (event.key === "Enter") { event.preventDefault(); input.blur(); }
        });
        input.addEventListener("wheel", event => event.preventDefault(), {passive: false});
      } else {
        input.textContent = String(counts[node.span]);
      }
      inputs.set(node.span, input);
      tree.append(input);
    }
  }

  function layout() {
    if (!args) return;
    const width = Math.max(viewport.clientWidth, args.links * 62 + 36);
    const maxDepth = Math.max(...args.nodes.map(node => node.depth));
    const height = Math.max(110, maxDepth * 76 + 72);
    tree.style.width = `${width}px`;
    tree.style.height = `${height}px`;
    const svg = tree.querySelector("svg");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    const positions = new Map();
    for (const node of args.nodes) {
      const x = args.links === 1 ? width / 2 : 36 + node.x * (width - 72) / (args.links - 1);
      const y = 36 + node.depth * 76;
      positions.set(node.span, {x, y});
      const input = inputs.get(node.span);
      input.style.left = `${x}px`;
      input.style.top = `${y}px`;
      if (args.editable) input.disabled = disabled;
    }
    for (const line of svg.children) {
      const parent = positions.get(line.dataset.parent), child = positions.get(line.dataset.child);
      line.setAttribute("x1", parent.x); line.setAttribute("y1", parent.y);
      line.setAttribute("x2", child.x); line.setAttribute("y2", child.y);
    }
    const frameHeight = height + (width > viewport.clientWidth ? 22 : 6);
    if (frameHeight !== lastHeight) {
      lastHeight = frameHeight;
      send("streamlit:setFrameHeight", {height: frameHeight});
    }
  }

  window.addEventListener("message", event => {
    if (event.source !== window.parent || event.data?.type !== "streamlit:render") return;
    const incoming = event.data.args;
    const changed = identity !== incoming.tree_id || args?.editable !== incoming.editable;
    args = incoming;
    disabled = Boolean(event.data.disabled);
    if (changed) {
      identity = incoming.tree_id;
      lastSent = null;
      counts = Object.fromEntries(args.nodes.map(node => [node.span, node.rounds]));
      rebuild();
    } else if (!lastSent || incoming.ack_id === lastSent) {
      // Do not overwrite a newer local edit with an older server acknowledgement.
      counts = Object.fromEntries(args.nodes.map(node => [node.span, node.rounds]));
      for (const node of args.nodes) {
        const input = inputs.get(node.span);
        if (args.editable) {
          if (document.activeElement !== input) input.value = String(node.rounds);
        } else input.textContent = String(node.rounds);
      }
    }
    tree.dataset.ackId = incoming.ack_id || "";
    layout();
  });
  new ResizeObserver(layout).observe(viewport);
  send("streamlit:componentReady", {apiVersion: 1});
})();
