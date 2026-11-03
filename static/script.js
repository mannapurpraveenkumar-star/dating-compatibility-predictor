// THE DATING GAME — front end. All results shown here come from the backend.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    form: $("upload-form"), input: $("file-input"), drop: $("dropzone"), dropEmpty: $("drop-empty"),
    preview: $("preview"), fileName: $("file-name"), analyse: $("analyse-btn"), change: $("change-btn"),
    status: $("status"), error: $("error"), limits: $("limits"),
    results: $("results"), count: $("count-line"), annotated: $("annotated"), caption: $("annotated-caption"),
    galleryBlock: $("gallery-block"), gallery: $("gallery"),
    matchList: $("match-list"), matchNote: $("match-note"),
    ageCheck: $("age-check"), ageList: $("age-list"), findPairs: $("find-pairs"),
    dlHtml: $("download-html"), dlMd: $("download-md"), again: $("again-btn"),
  };

  const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
  let maxBytes = 10 * 1024 * 1024;
  let file = null;
  let previewUrl = null;
  let lastResult = null;

  // ---- helpers -------------------------------------------------------------
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") node.className = v;
      else if (k === "style") node.setAttribute("style", v);
      else if (k === "dataset") Object.assign(node.dataset, v);
      else node[k] = v;
    }
    for (const c of children) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    return node;
  }

  function setStatus(text, busy = false) {
    els.status.textContent = text;
    els.status.classList.toggle("busy", busy);
  }

  function showError(text) {
    els.error.textContent = text;
    els.error.hidden = !text;
  }

  function formatBytes(n) {
    return n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`;
  }

  function personChip(label, colour) {
    return el("span", { class: "people", style: `--c:${colour}` }, el("span", { class: "swatch" }), label);
  }

  // ---- limits from the server --------------------------------------------
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      if (h.limits) {
        maxBytes = h.limits.max_upload_mb * 1024 * 1024;
        els.limits.textContent = `JPEG, PNG or WebP, up to ${h.limits.max_upload_mb} MB`;
      }
      if (!h.detector_ready) showError(`The face detector isn't ready: ${h.detector_error}`);
    })
    .catch(() => showError("Can't reach the server. Check that it is running, then reload this page."));

  // ---- choosing a file ----------------------------------------------------
  function chooseFile(f) {
    showError("");
    if (!f) return;
    if (!ACCEPTED.includes(f.type)) {
      showError(`“${f.name}” isn't a JPEG, PNG or WebP image. Choose a photo in one of those formats.`);
      return;
    }
    if (f.size > maxBytes) {
      showError(`“${f.name}” is ${formatBytes(f.size)}. The limit is ${formatBytes(maxBytes)}.`);
      return;
    }
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(f);
    els.preview.src = previewUrl;
    els.preview.hidden = false;
    els.dropEmpty.hidden = true;
    els.fileName.textContent = `${f.name} (${formatBytes(f.size)})`;
    els.fileName.hidden = false;
    els.analyse.disabled = false;
    els.change.hidden = false;
    setStatus("Ready. Press Analyse Photo.");
  }

  els.input.addEventListener("change", () => chooseFile(els.input.files[0]));
  els.change.addEventListener("click", () => els.input.click());

  ["dragenter", "dragover"].forEach((t) =>
    els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((t) =>
    els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("dragging"); }));
  els.drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) chooseFile(f);
  });
  // Stop the browser opening a photo dropped just outside the zone.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // ---- analyse ------------------------------------------------------------
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    run();
  });

  async function run(answers = null) {
    if (!file) return;
    showError("");
    els.analyse.disabled = true;
    els.change.disabled = true;
    els.findPairs.disabled = true;
    setStatus(answers ? "Finding pairs with your answers…" : "Finding faces and measuring the photo…", true);

    const body = new FormData();
    body.append("photo", file, file.name);
    if (answers) {
      body.append("adults", answers.adults.join(","));
      body.append("children", answers.children.join(","));
    }
    try {
      const res = await fetch("/api/analyse", { method: "POST", body });
      let data = null;
      try { data = await res.json(); } catch { /* non-JSON error page */ }
      if (!res.ok) {
        const msg = data && data.detail && (data.detail.message || data.detail);
        throw new Error(typeof msg === "string" ? msg : `The server returned an error (${res.status}).`);
      }
      lastResult = data;
      render(data);
      setStatus("Analysis complete.");
    } catch (err) {
      const msg = err instanceof TypeError ? "Can't reach the server. Check that it is running." : err.message;
      showError(msg);
      setStatus("");
    } finally {
      els.analyse.disabled = !file;
      els.change.disabled = false;
      els.findPairs.disabled = false;
    }
  }

  // "Check ages": the user's answers override the models, which can't tell
  // children from adults on small faces.
  function renderAgeCheck(r) {
    const unreadable = (age) => ["face too small to check age", "face too blurry to check age",
      "age could not be checked"].includes(age.reason);
    els.ageList.replaceChildren(...r.faces.map((f, i) => {
      const age = f.features.age || {};
      const crop = r.crops[i];
      const name = `age-${f.letter}`;
      const preset = age.confirmed_by_user ? (age.adult ? "adult" : "child")
        : unreadable(age) ? null : (age.adult ? "adult" : "child");
      const radio = (value, text) => {
        const input = el("input", { type: "radio", name, value });
        if (preset === value) input.checked = true;
        return el("label", { class: "age-opt" }, input, text);
      };
      const hint = age.confirmed_by_user ? "Your answer"
        : unreadable(age) ? "Too small for the app to check" : age.adult ? "App: adult" : "App: may be under 18";
      return el("li", { dataset: { label: f.label } },
        el("img", { src: crop.image, alt: `Face of ${f.label}` }),
        el("div", {}, el("strong", {}, f.label), el("div", { class: "meta" }, hint),
          el("div", { class: "age-opts", role: "radiogroup", ariaLabel: `Age group of ${f.label}` },
            radio("adult", "Adult"), radio("child", "Under 18"))));
    }));
    els.ageCheck.hidden = r.face_count < 2;
    els.ageCheck.open = Boolean(r.prediction.needs_age_check);
  }

  els.findPairs.addEventListener("click", () => {
    const answers = { adults: [], children: [] };
    for (const li of els.ageList.children) {
      const picked = li.querySelector("input:checked");
      if (picked) answers[picked.value === "adult" ? "adults" : "children"].push(li.dataset.label);
    }
    run(answers);
  });

  // Points each feature added to a pair's score, so no single feature looks like the whole reason.
  function breakdown(rows) {
    const measured = rows.filter((b) => b.points !== null);
    const missing = rows.filter((b) => b.points === null).map((b) => b.name.toLowerCase());
    const list = el("ul", { class: "breakdown", ariaLabel: "How the score was made" },
      ...measured.map((b) => el("li", {},
        el("span", { class: "bd-name" }, b.name),
        el("span", { class: "bd-bar", ariaHidden: "true" },
          el("span", { class: "bd-max", style: `width:${b.max_points}%` },
            el("span", { class: "bd-fill", style: `width:${b.max_points ? (100 * b.points) / b.max_points : 0}%` }))),
        el("span", { class: "bd-pts" }, `+${Math.round(b.points)} of ${Math.round(b.max_points)}`))));
    const wrap = el("div", {}, list);
    if (missing.length) wrap.append(el("p", { class: "bd-note" }, `Not measurable in this photo: ${missing.join(", ")}.`));
    return wrap;
  }

  // ---- render -------------------------------------------------------------
  function render(r) {
    const n = r.face_count;
    const p = r.prediction;
    els.count.replaceChildren();
    if (n === 0) els.count.append("No faces found.");
    else {
      els.count.append(el("strong", {}, `${n} ${n === 1 ? "face" : "faces"}`), " found");
      if (n >= 2) els.count.append(", ", el("strong", {}, `${r.pair_count} pairs`), " compared");
      els.count.append(".");
    }

    // one card per predicted pair
    const crops = Object.fromEntries(r.crops.map((c) => [c.label, c]));
    const cards = p.pairs.map((x, i) => {
      const [a, b] = x.pair;
      return el("article", { class: "match-card" + (i ? " secondary" : "") },
        el("div", { class: "match-faces" },
          el("img", { src: crops[a].image, alt: `Face of ${a}` }),
          el("span", { class: "match-heart", ariaHidden: "true" }, "♥"),
          el("img", { src: crops[b].image, alt: `Face of ${b}` })),
        el("div", { class: "match-text" },
          el("p", { class: "match-names" }, `${a} & ${b}`),
          el("p", { class: "match-score" },
            el("span", { class: "score-num" }, x.match_score), el("span", { class: "score-max" }, "/100"),
            el("span", { class: "stamp" }, x.level)),
          breakdown(x.breakdown)));
    });
    if (!cards.length) {
      cards.push(el("article", { class: "match-card none" },
        el("div", { class: "match-text" }, el("p", { class: "match-names" }, "No pair predicted"))));
    }
    els.matchList.replaceChildren(...cards);
    const notes = [p.message, "Every person’s result is listed under “Everyone in the photo”."];
    if (p.pairs.length) notes.push("This is an experimental prediction, not proof that anyone is a couple.");
    els.matchNote.textContent = notes.join(" ");

    // photo
    els.annotated.src = r.annotated_image;
    els.annotated.alt = p.pairs.length
      ? `Your photo with ${n} faces outlined. Predicted pairs highlighted: ${p.pairs.map((x) => x.pair.join(" and ")).join("; ")}.`
      : n ? `Your photo with ${n} detected ${n === 1 ? "face" : "faces"} outlined.` : "Your photo. No faces were detected.";
    els.caption.textContent = p.pairs.length
      ? "Predicted pairs are highlighted in pink and joined by a line. People are labelled from left to right."
      : n ? "Boxes show where faces were found. People are labelled from left to right."
          : "No faces found. Faces may be too small, turned away, covered or poorly lit.";

    // gallery: every person's result, so nobody is left unexplained
    const status = Object.fromEntries((p.people_status || []).map((s) => [s.label, s]));
    els.gallery.replaceChildren(...r.crops.map((c) => {
      const st = status[c.label] || { status: "", detail: "" };
      return el("li", { class: `${c.in_pair ? "in-pair" : ""} st-${st.status}`, style: `--c:${c.colour}` },
        el("img", { src: c.image, alt: `Face crop of ${c.label}` }),
        el("div", { class: "who" }, el("span", { class: "swatch" }), c.label),
        el("div", { class: "meta" }, st.detail));
    }));
    els.galleryBlock.hidden = n === 0;

    renderAgeCheck(r);
    els.results.hidden = false;
    els.results.focus({ preventScroll: true });
    els.results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }

  // ---- downloads ----------------------------------------------------------
  function download(text, type, ext) {
    const stamp = (lastResult.generated_at || "").replace(/[:T]/g, "-").slice(0, 16);
    const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: `dating-game-report-${stamp}.${ext}` });
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }
  els.dlHtml.addEventListener("click", () => lastResult && download(lastResult.report_html, "text/html", "html"));
  els.dlMd.addEventListener("click", () => lastResult && download(lastResult.report_markdown, "text/markdown", "md"));

  // ---- start again --------------------------------------------------------
  els.again.addEventListener("click", () => {
    file = null;
    lastResult = null;
    els.form.reset();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    els.preview.hidden = true;
    els.preview.removeAttribute("src");
    els.dropEmpty.hidden = false;
    els.fileName.hidden = true;
    els.analyse.disabled = true;
    els.change.hidden = true;
    els.results.hidden = true;
    els.annotated.removeAttribute("src");
    showError("");
    setStatus("");
    window.scrollTo({ top: 0 });
    els.input.focus();
  });
})();
