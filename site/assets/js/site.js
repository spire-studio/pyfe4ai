/* PyFE4AI project page: theme toggle, in-view animation gates, diagram players,
   code tabs and copy buttons. No dependencies, no network. */
(function () {
  "use strict";
  var root = document.documentElement;
  var reduce = false;
  try { reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { /* old browsers */ }
  var SVGNS = "http://www.w3.org/2000/svg";
  var KEY = "pyfe4ai-theme";

  /* ---------- theme ---------- */
  function systemTheme() {
    try { return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"; } catch (e) { return "dark"; }
  }
  function currentTheme() { return root.getAttribute("data-theme") || systemTheme(); }
  document.querySelectorAll(".theme-toggle").forEach(function (btn) {
    function label() {
      btn.setAttribute("aria-label", currentTheme() === "dark" ? "Switch to the light theme" : "Switch to the dark theme");
    }
    label();
    btn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      // a choice equal to the system's is no choice: follow the system again
      if (next === systemTheme()) {
        root.removeAttribute("data-theme");
        try { localStorage.removeItem(KEY); } catch (e) { /* storage unavailable */ }
      } else {
        root.setAttribute("data-theme", next);
        try { localStorage.setItem(KEY, next); } catch (e) { /* storage unavailable */ }
      }
      label();
    });
  });

  /* ---------- mobile menu: close after a jump ---------- */
  document.querySelectorAll(".nav-menu").forEach(function (menu) {
    menu.addEventListener("click", function (ev) {
      if (ev.target.closest("a")) menu.removeAttribute("open");
    });
    document.addEventListener("click", function (ev) {
      if (menu.hasAttribute("open") && !menu.contains(ev.target)) menu.removeAttribute("open");
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && menu.hasAttribute("open")) { menu.removeAttribute("open"); menu.querySelector("summary").focus(); }
    });
  });

  /* ---------- in-view: reveal once, run animations only while visible ---------- */
  var hasIO = "IntersectionObserver" in window;
  var gates = document.querySelectorAll(".anim-gate");
  var reveals = document.querySelectorAll(".reveal");
  if (hasIO) {
    var gateIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        en.target.classList.toggle("in-view", en.isIntersecting);
        if (en.isIntersecting) en.target.classList.add("was-seen");
        var p = en.target.__player;
        if (p) p.visible(en.isIntersecting);
      });
    }, { threshold: 0.15 });
    gates.forEach(function (g) { gateIO.observe(g); });
    var revIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add("is-visible"); revIO.unobserve(en.target); }
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -5% 0px" });
    reveals.forEach(function (r) { revIO.observe(r); });
  } else {
    gates.forEach(function (g) { g.classList.add("in-view", "was-seen"); });
    reveals.forEach(function (r) { r.classList.add("is-visible"); });
  }

  /* ---------- diagram player ---------- */
  function Player(el, scenarios) {
    var svg = el.querySelector("svg.dg");
    var pk = svg.querySelector(".packets");
    var tabs = el.querySelectorAll(".tab");
    var lists = el.querySelectorAll(".steps");
    var playBtn = el.querySelector(".play");
    var scn = tabs.length ? tabs[0].getAttribute("data-scn") : Object.keys(scenarios)[0];
    var idx = 0, playing = !reduce, inView = !hasIO, timer = null, token = 0;

    function steps() { return scenarios[scn]; }
    function list() {
      var l = null;
      lists.forEach(function (o) { if (o.getAttribute("data-scn") === scn) l = o; });
      return l;
    }
    function setVariant() {
      svg.querySelectorAll("[data-v]").forEach(function (n) {
        n.classList.toggle("off", n.getAttribute("data-v").split(" ").indexOf(scn) < 0);
      });
      lists.forEach(function (o) { o.hidden = o.getAttribute("data-scn") !== scn; });
    }
    function clear() {
      svg.querySelectorAll(".is-on").forEach(function (n) { n.classList.remove("is-on"); });
      while (pk.firstChild) pk.removeChild(pk.firstChild);
    }
    function on(ids) {
      (ids || []).forEach(function (id) {
        var n = svg.querySelector('[data-id="' + id + '"]');
        if (n) n.classList.add("is-on");
      });
    }
    function follow(ids) {
      // a diagram wider than its box scrolls inside it: bring the step's nodes into view
      var box = svg.parentNode;
      if (!box || el.hasAttribute("data-nofollow") || box.scrollWidth <= box.clientWidth + 2 || !ids || !ids.length) return;
      var n = svg.querySelector('[data-id="' + ids[ids.length - 1] + '"]');
      if (!n) return;
      var r = n.getBoundingClientRect(), b = box.getBoundingClientRect();
      var left = box.scrollLeft + (r.left + r.width / 2) - (b.left + b.width / 2);
      try { box.scrollTo({ left: left, behavior: reduce ? "auto" : "smooth" }); } catch (e) { box.scrollLeft = left; }
    }
    function packet(edge, my) {
      return new Promise(function (resolve) {
        var path = svg.querySelector('[data-edge="' + edge.id + '"]');
        if (!path) { resolve(); return; }
        path.classList.add("is-on");
        var len = path.getTotalLength();
        var up = path.classList.contains("up") ? " up" : "";
        if (edge.solo) while (pk.firstChild) pk.removeChild(pk.firstChild);
        var g = document.createElementNS(SVGNS, "g");
        var halo = document.createElementNS(SVGNS, "circle");
        halo.setAttribute("r", "12"); halo.setAttribute("class", "pkt-halo" + up);
        var dot = document.createElementNS(SVGNS, "circle");
        dot.setAttribute("r", "5.5"); dot.setAttribute("class", "pkt" + up);
        g.appendChild(halo); g.appendChild(dot);
        var tx = null;
        if (edge.label) {
          tx = document.createElementNS(SVGNS, "text");
          tx.setAttribute("class", "pkt-label"); tx.setAttribute("x", "10"); tx.setAttribute("y", "-10");
          tx.textContent = edge.label; g.appendChild(tx);
        }
        pk.appendChild(g);
        function place(f) {
          var p = path.getPointAtLength(edge.rev ? len * (1 - f) : len * f);
          g.setAttribute("transform", "translate(" + p.x.toFixed(1) + "," + p.y.toFixed(1) + ")");
          // the label fades before the packet reaches its node, so that it never covers the node's title
          if (tx) tx.style.opacity = f < 0.75 ? "1" : String(Math.max(0, (1 - f) / 0.25));
        }
        if (reduce) { place(1); resolve(); return; }
        var dur = edge.solo ? Math.max(260, len * 4.2) : Math.max(600, Math.min(1500, len * 3)), start = null;
        function tick(now) {
          if (my !== token) { resolve(); return; }
          if (start === null) start = now;
          var f = Math.min(1, (now - start) / dur);
          var e = f < 0.5 ? 2 * f * f : 1 - Math.pow(-2 * f + 2, 2) / 2;
          place(e);
          if (f < 1) requestAnimationFrame(tick); else resolve();
        }
        requestAnimationFrame(tick);
      });
    }
    function show(k, announce) {
      token++; var my = token;
      clearTimeout(timer);
      clear();
      var s = steps()[k];
      on(s.nodes);
      follow(s.after && s.after.length ? s.after : s.nodes);
      var l = list();
      if (l) {
        l.setAttribute("aria-live", announce ? "polite" : "off");
        l.querySelectorAll("li").forEach(function (li, i) {
          li.classList.toggle("is-on", i === k);
          if (i === k) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
        });
      }
      // a step's label goes on its first packet only
      var edges = (s.edges || []).map(function (e, i) {
        var o = typeof e === "string" ? { id: e, label: i === 0 ? s.label : "" } : e;
        if (s.seq) { o.solo = true; if (!o.label) o.label = s.label; }
        return o;
      });
      var run = s.seq
        ? edges.reduce(function (p, e) { return p.then(function () { return my === token ? packet(e, my) : null; }); }, Promise.resolve())
        : Promise.all(edges.map(function (e) { return packet(e, my); }));
      run.then(function () {
        if (my !== token) return;
        on(s.after);
        schedule(k === steps().length - 1 ? 2800 : 1700);
      });
    }
    function schedule(ms) {
      clearTimeout(timer);
      if (!playing || !inView || document.hidden) return;
      timer = setTimeout(function () { idx = (idx + 1) % steps().length; show(idx, false); }, ms);
    }
    function setPlaying(p) {
      playing = p;
      if (playBtn) {
        playBtn.setAttribute("aria-label", p ? "Pause" : "Play");
        playBtn.querySelector("use").setAttribute("href", p ? "#i-pause" : "#i-play");
      }
      if (p) schedule(400); else clearTimeout(timer);
    }
    function go(k) { setPlaying(false); idx = (k + steps().length) % steps().length; show(idx, true); }
    tabs.forEach(function (b) {
      b.addEventListener("click", function () {
        tabs.forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        scn = b.getAttribute("data-scn"); idx = 0; setVariant(); show(0, true);
      });
    });
    lists.forEach(function (l) {
      l.querySelectorAll("li").forEach(function (li, i) {
        li.addEventListener("click", function () { if (l.getAttribute("data-scn") === scn) go(i); });
      });
    });
    var prev = el.querySelector(".prev"), next = el.querySelector(".next");
    if (prev) prev.addEventListener("click", function () { go(idx - 1); });
    if (next) next.addEventListener("click", function () { go(idx + 1); });
    if (playBtn) playBtn.addEventListener("click", function () { setPlaying(!playing); });
    document.addEventListener("visibilitychange", function () { if (!document.hidden) schedule(600); });
    el.__player = {
      visible: function (v) { inView = v; if (v) schedule(500); else clearTimeout(timer); }
    };
    setVariant();
    setPlaying(playing);
    show(0, false);
  }

  var cAll = ["c1", "c2", "c3"];
  var players = {
    fl: {
      mcfe: [
        { nodes: ["auth"], edges: ["a-c1", "a-c2", "a-c3"], label: "skᵢ", after: cAll },
        { nodes: cAll, edges: ["c1-agg", "c2-agg", "c3-agg"], label: "ct", after: ["agg"] },
        { nodes: ["auth"], edges: [{ id: "a-agg", label: "dk_y" }], after: ["agg"] },
        { nodes: ["agg"], edges: [{ id: "agg-res", label: "Σ" }], after: ["res"] }
      ],
      thr: [
        { nodes: ["auth"], edges: [{ id: "a-c1", label: "skᵢ" }, { id: "a-c2" }, { id: "a-c3" }, { id: "a-s1", label: "share" }, { id: "a-s2" }, { id: "a-s3" }],
          after: ["c1", "c2", "c3", "s1", "s2", "s3"] },
        { nodes: cAll, edges: ["c1-s1", "c1-s2", "c1-s3", "c2-s1", "c2-s2", "c2-s3", "c3-s1", "c3-s2", "c3-s3"], after: ["s1", "s2", "s3"] },
        { nodes: ["s1", "s2"], edges: [{ id: "s1-comb", label: "partial" }, { id: "s2-comb" }], after: ["comb"] },
        { nodes: ["comb"], edges: [{ id: "agg-res", label: "Σ" }], after: ["res"] }
      ],
      dec: [
        { nodes: ["joint"], edges: ["a-c1", "a-c2", "a-c3"], label: "keys", after: cAll },
        { nodes: cAll, edges: [{ id: "c1-agg", label: "ct + dkᵢ" }, { id: "c2-agg" }, { id: "c3-agg" }], after: ["aggd"] },
        { nodes: ["aggd"], edges: [{ id: "agg-res", label: "Σ" }], after: ["res"] }
      ]
    }
  };
  var upFromBe = ["u4", "u3", "u2", "u1", "u0"];
  players.arch = {
    create: [
      { nodes: ["app", "core"], edges: ["d0", "d1", "u1", "u0"], seq: true, after: ["app"] },
      { nodes: ["app", "sch"], edges: ["d0", "d1", "d2"], seq: true },
      { nodes: ["prim"], edges: ["d3"], seq: true },
      { nodes: ["be"], edges: ["d4"], seq: true },
      { nodes: ["rust"], edges: ["dr"], seq: true },
      { nodes: ["app"], edges: ["ur"].concat(upFromBe), seq: true }
    ],
    ml: [
      { nodes: ["app"] },
      { nodes: ["ml"], edges: ["d0"], seq: true },
      { nodes: ["sch"], edges: ["d1", "d2"], seq: true },
      { nodes: ["prim", "be", "rust"], edges: ["d3", "d4", "dr"], seq: true },
      { nodes: ["app"], edges: ["ur"].concat(upFromBe), seq: true }
    ],
    py: [
      { nodes: ["app", "be"] },
      { nodes: ["sch", "prim"], edges: ["d0", "d1", "d2", "d3"], seq: true },
      { nodes: ["be", "py"], edges: ["d4", "dp"], seq: true },
      { nodes: ["app"], edges: ["up"].concat(upFromBe), seq: true }
    ]
  };

  document.querySelectorAll("[data-player]").forEach(function (el) {
    var conf = players[el.getAttribute("data-player")];
    if (conf && el.querySelector("svg.dg")) Player(el, conf);
  });

  /* ---------- scheme matrix: list the schemes of a cell ---------- */
  var STATUS_CLASS = { "reviewed": "rev", "experimental": "exp", "unreviewed": "unr", "known-insecure": "ins" };
  document.querySelectorAll(".matrix-card").forEach(function (card) {
    var detail = card.querySelector(".cell-detail");
    var cells = card.querySelectorAll(".cell");
    function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
    cells.forEach(function (c) {
      c.setAttribute("aria-pressed", "false");
      c.addEventListener("click", function () {
        cells.forEach(function (o) { o.setAttribute("aria-pressed", o === c ? "true" : "false"); });
        var items = c.getAttribute("data-schemes").split(";").map(function (t) {
          var f = t.split("|");
          return '<li><code>' + esc(f[0]) + '</code><span class="badge ' + (STATUS_CLASS[f[1]] || "unr") + '">' + esc(f[1]) + "</span>" +
            (f[2] ? '<span class="badge ref">refused under standard</span>' : "") + "</li>";
        });
        detail.innerHTML = "<h4>" + esc(c.getAttribute("data-cell")) + "</h4><ul>" + items.join("") + "</ul>";
      });
    });
  });

  /* ---------- code tabs (WAI-ARIA tabs, manual activation) ---------- */
  document.querySelectorAll(".code-tabs").forEach(function (box) {
    var tabs = Array.prototype.slice.call(box.querySelectorAll('[role="tab"]'));
    function select(t, focus) {
      tabs.forEach(function (o) {
        var on = o === t;
        o.setAttribute("aria-selected", on ? "true" : "false");
        o.tabIndex = on ? 0 : -1;
        var panel = document.getElementById(o.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      });
      if (focus) t.focus();
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t, false); });
      t.addEventListener("keydown", function (ev) {
        var k = ev.key, j = null;
        if (k === "ArrowRight") j = (i + 1) % tabs.length;
        else if (k === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
        else if (k === "Home") j = 0;
        else if (k === "End") j = tabs.length - 1;
        if (j !== null) { ev.preventDefault(); select(tabs[j], true); }
      });
    });
    select(tabs[0], false);
  });

  /* ---------- nav: mark the section in view ---------- */
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.nav-links a[href^="#"]'));
  if (hasIO && navLinks.length) {
    var byId = {};
    navLinks.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var a = byId[en.target.id];
        if (!a) return;
        if (en.isIntersecting) {
          navLinks.forEach(function (o) { o.removeAttribute("aria-current"); });
          a.setAttribute("aria-current", "true");
        } else if (a.getAttribute("aria-current")) {
          a.removeAttribute("aria-current");
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    Object.keys(byId).forEach(function (id) { var sec = document.getElementById(id); if (sec) spy.observe(sec); });
  }

  /* ---------- copy buttons ---------- */
  document.querySelectorAll(".code").forEach(function (box) {
    var head = box.querySelector(".code-head");
    if (!head || head.querySelector(".copy")) return;
    var btn = document.createElement("button");
    btn.type = "button"; btn.className = "copy";
    btn.innerHTML = '<svg aria-hidden="true"><use href="#i-copy"/></svg><span>Copy</span>';
    head.appendChild(btn);
    btn.addEventListener("click", function () {
      var pre = box.querySelector(".cpanel:not([hidden]) pre") || box.querySelector("pre");
      var text = pre.innerText.replace(/\n$/, "");
      function done(ok) {
        btn.querySelector("span").textContent = ok ? "Copied" : "Press Ctrl+C";
        btn.querySelector("use").setAttribute("href", ok ? "#i-check" : "#i-copy");
        setTimeout(function () {
          btn.querySelector("span").textContent = "Copy";
          btn.querySelector("use").setAttribute("href", "#i-copy");
        }, 1800);
      }
      function fallback() {
        var r = document.createRange(); r.selectNodeContents(pre);
        var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        var ok = false;
        try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
        done(ok);
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function () { done(true); }, fallback);
      } else {
        fallback();
      }
    });
  });
})();
