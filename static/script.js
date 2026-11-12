// THE DATING GAME — front end. All results shown here come from the backend.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    form: $("upload-form"), input: $("file-input"), drop: $("dropzone"), dropEmpty: $("drop-empty"),
    strip: $("preview-strip"), fileName: $("file-name"), analyse: $("analyse-btn"), change: $("change-btn"),
    status: $("status"), error: $("error"), limits: $("limits"),
    results: $("results"), count: $("count-line"), annotatedList: $("annotated-list"),
    galleryBlock: $("gallery-block"), gallery: $("gallery"),
    matchList: $("match-list"), matchNote: $("match-note"),
    weightsPanel: $("weights-panel"), weightList: $("weight-list"), recalc: $("recalc"),
    presetSimilarity: $("preset-similarity"), presetPosition: $("preset-position"),
    ratingsPanel: $("ratings-panel"), peopleTable: $("people-table"), pairsTable: $("pairs-table"), attrNote: $("attr-note"),
    ageCheck: $("age-check"), ageList: $("age-list"), findPairs: $("find-pairs"),
    dlHtml: $("download-html"), dlMd: $("download-md"), again: $("again-btn"),
  };

  const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
  let maxBytes = 10 * 1024 * 1024;
  let maxPhotos = 5;
  let files = [];
  let previewUrls = [];
  let lastResult = null;
  let factors = [];          // [{key, name, group, default_weight, description}]
  let weights = null;        // {key: 0-100}; null until the factor list has loaded
  let answers = null;        // the user's "Check ages" answers, kept for recalculations

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

  const pct = (v) => (v === null || v === undefined ? "–" : String(Math.round(v * 100)));

  // ---- limits and factors from the server --------------------------------
  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      if (h.limits) {
        maxBytes = h.limits.max_upload_mb * 1024 * 1024;
        maxPhotos = h.limits.max_photos || maxPhotos;
        els.limits.textContent = `JPEG, PNG or WebP, up to ${maxPhotos} photos of ${h.limits.max_upload_mb} MB each`;
      }
      if (!h.detector_ready) showError(`The face detector isn't ready: ${h.detector_error}`);
    })
    .catch(() => showError("Can't reach the server. Check that it is running, then reload this page."));

  fetch("/api/factors")
    .then((r) => r.json())
    .then((d) => {
      factors = d.factors;
      weights = Object.fromEntries(factors.map((f) => [f.key, f.default_weight]));
      renderWeights();
    })
    .catch(() => { /* sliders stay empty; the server's defaults apply */ });

  // ---- choosing photos ----------------------------------------------------
  function chooseFiles(list) {
    showError("");
    const picked = Array.from(list || []);
    if (!picked.length) return;
    if (picked.length > maxPhotos) {
      showError(`Choose at most ${maxPhotos} photos at once.`);
      return;
    }
    for (const f of picked) {
      if (!ACCEPTED.includes(f.type)) {
        showError(`“${f.name}” isn't a JPEG, PNG or WebP image. Choose photos in one of those formats.`);
        return;
      }
      if (f.size > maxBytes) {
        showError(`“${f.name}” is ${formatBytes(f.size)}. The limit is ${formatBytes(maxBytes)} per photo.`);
        return;
      }
    }
    files = picked;
    answers = null;
    previewUrls.forEach((u) => URL.revokeObjectURL(u));
    previewUrls = files.map((f) => URL.createObjectURL(f));
    els.strip.replaceChildren(...previewUrls.map((u, i) =>
      el("img", { src: u, alt: `Preview of photo ${i + 1}`, class: files.length > 1 ? "thumb" : "single" })));
    els.strip.hidden = false;
    els.dropEmpty.hidden = true;
    els.fileName.textContent = files.length === 1
      ? `${files[0].name} (${formatBytes(files[0].size)})`
      : `${files.length} photos: faces from all of them form one group`;
    els.fileName.hidden = false;
    els.analyse.disabled = false;
    els.analyse.textContent = files.length > 1 ? "Analyse Photos" : "Analyse Photo";
    els.change.hidden = false;
    setStatus(`Ready. Press ${els.analyse.textContent}.`);
  }

  els.input.addEventListener("change", () => chooseFiles(els.input.files));
  els.change.addEventListener("click", () => els.input.click());

  ["dragenter", "dragover"].forEach((t) =>
    els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((t) =>
    els.drop.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("dragging"); }));
  els.drop.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length) chooseFiles(e.dataTransfer.files);
  });
  // Stop the browser opening a photo dropped just outside the zone.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  // ---- analyse ------------------------------------------------------------
  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    answers = null;
    run("Finding faces and measuring the photos…");
  });

  async function run(message) {
    if (!files.length) return;
    showError("");
    for (const b of [els.analyse, els.change, els.findPairs, els.recalc]) b.disabled = true;
    setStatus(message, true);

    const body = new FormData();
    for (const f of files) body.append("photo", f, f.name);
    if (answers) {
      body.append("adults", answers.adults.join(","));
      body.append("children", answers.children.join(","));
    }
    if (weights) body.append("weights", JSON.stringify(weights));
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
      els.analyse.disabled = !files.length;
      for (const b of [els.change, els.findPairs, els.recalc]) b.disabled = false;
    }
  }

  // ---- importance of each factor -------------------------------------------
  function renderWeights() {
    if (!weights) return;
    const groups = [["similarity", "Who they are"], ["proximity", "Where they stand (only within one photo)"]];
    els.weightList.replaceChildren(...groups.map(([group, title]) => el("fieldset", { class: "weight-group" },
      el("legend", {}, title),
      ...factors.filter((f) => f.group === group).map((f) => {
        const id = `w-${f.key}`;
        const out = el("output", { htmlFor: id, class: "w-val" }, weights[f.key]);
        const input = el("input", { type: "range", id, min: 0, max: 100, step: 5, value: weights[f.key] });
        input.addEventListener("input", () => {
          weights[f.key] = Number(input.value);
          out.textContent = input.value;
        });
        return el("div", { class: "weight-row", title: f.description },
          el("label", { htmlFor: id }, f.name), input, out);
      }))));
  }

  function preset(withPosition) {
    if (!weights) return;
    for (const f of factors) weights[f.key] = f.default_weight;
    if (withPosition) Object.assign(weights, { closeness: 20, side_by_side: 10, leaning_in: 10 });
    renderWeights();
  }
  els.presetSimilarity.addEventListener("click", () => preset(false));
  els.presetPosition.addEventListener("click", () => preset(true));
  els.recalc.addEventListener("click", () => run("Recalculating with your weights…"));

  // ---- "Check ages": the user's answers override the models ----------------
  function renderAgeCheck(r) {
    const unreadable = (age) => ["face too small to check age", "face too blurry to check age",
      "age could not be checked"].includes(age.reason);
    const cropOf = Object.fromEntries(r.crops.map((c) => [c.label, c]));
    els.ageList.replaceChildren(...r.faces.filter((f) => !f.features.duplicate_of).map((f) => {
      const age = f.features.age || {};
      const crop = cropOf[f.label];
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
    answers = { adults: [], children: [] };
    for (const li of els.ageList.children) {
      const picked = li.querySelector("input:checked");
      if (picked) answers[picked.value === "adult" ? "adults" : "children"].push(li.dataset.label);
    }
    run("Finding pairs with your answers…");
  });

  // ---- score breakdown on each match card -----------------------------------
  function breakdown(rows, uninstalled = []) {
    const measured = rows.filter((b) => b.points !== null);
    const missing = rows.filter((b) => b.status === "not measurable" && !uninstalled.includes(b.feature))
      .map((b) => b.name.toLowerCase());
    const list = el("ul", { class: "breakdown", ariaLabel: "How the score was made" },
      ...measured.map((b) => el("li", {},
        el("span", { class: "bd-name" }, b.name),
        el("span", { class: "bd-bar", ariaHidden: "true" },
          el("span", { class: "bd-max", style: `width:${b.max_points}%` },
            el("span", { class: "bd-fill", style: `width:${b.max_points ? (100 * b.points) / b.max_points : 0}%` }))),
        el("span", { class: "bd-pts" }, `+${Math.round(b.points)} of ${Math.round(b.max_points)}`))));
    const wrap = el("div", {}, list);
    if (missing.length) wrap.append(el("p", { class: "bd-note" }, `Not measurable for this pair: ${missing.join(", ")}.`));
    return wrap;
  }

  // ---- all ratings tables ------------------------------------------------------
  function table(node, header, rows) {
    node.replaceChildren(
      el("thead", {}, el("tr", {}, ...header.map((h) => el("th", { scope: "col" }, h)))),
      el("tbody", {}, ...rows));
  }

  function renderRatings(r) {
    const crops = Object.fromEntries(r.crops.map((c) => [c.label, c]));
    const hasAttr = r.faces.some((f) => f.features.attractiveness);
    const multi = r.photo_count > 1;
    table(els.peopleTable,
      ["Person", ...(multi ? ["Photo"] : []), "Age (est.)", "Apparent gender", "Attractiveness (1–5)", "Outfit", "Adult"],
      r.faces.map((f) => {
        const age = f.features.age || {};
        const g = age.gender;
        const a = f.features.attractiveness;
        const o = f.features.outfit || {};
        const adult = f.features.duplicate_of ? `Same person as ${f.features.duplicate_of}`
          : age.adult ? "Yes" : `No: ${age.reason || "unknown"}`;
        return el("tr", {},
          el("td", {}, el("span", { class: "rt-person" },
            el("img", { src: crops[f.label].image, alt: "" }), f.label)),
          ...(multi ? [el("td", {}, f.photo)] : []),
          el("td", { class: "num" }, age.years ?? "–"),
          el("td", {}, g ? `${g.apparent} (${Math.round(g.confidence * 100)}%)` : "–"),
          a ? el("td", { class: "num" }, a.score.toFixed(2) + (a.reliable ? "" : " *"))
            : el("td", { class: "muted" }, "not installed"),
          el("td", {}, o.visible ? el("span", { class: "rt-colour" },
            el("span", { class: "swatch", style: `--c:${o.hex}` }), o.name) : "not visible"),
          el("td", {}, adult + (age.confirmed_by_user ? " (your answer)" : "")));
      }));
    els.attrNote.textContent = hasAttr
      ? "Attractiveness comes from a model trained on 5,500 faces (73% East Asian, 27% white) rated by 60 volunteers. " +
        "It reflects their taste, not a fact, and is unreliable for other groups. * = face too small to rate reliably."
      : "Attractiveness ratings appear once the model is installed (see the handover document).";

    const chosen = new Set(r.prediction.pairs.map((x) => x.pair.join("|")));
    const excluded = new Set(r.prediction.excluded_people.map((x) => x.label));
    const dupOf = Object.fromEntries(r.faces.filter((f) => f.features.duplicate_of)
      .map((f) => [f.label, f.features.duplicate_of]));
    const status = (p) => chosen.has(`${p.a}|${p.b}`) || chosen.has(`${p.b}|${p.a}`) ? "Predicted"
      : dupOf[p.a] === p.b || dupOf[p.b] === p.a ? "Same person"
      : dupOf[p.a] || dupOf[p.b] ? "Repeat of a person (counted once)"
      : excluded.has(p.a) || excluded.has(p.b) ? "Person not matched"
        : p.blocked ? "Same apparent gender"
          : p.match_score < (r.min_match_score ?? 35) ? "Below cut-off" : "Other pair fits better";
    table(els.pairsTable,
      ["Pair", "Score", ...r.factors.map((f) => `${f.name}${r.weights[f.key] ? "" : " (off)"}`), "Status"],
      // pairs that can be matched first (by score), then pairs with someone not matched
      [...r.pair_scores].sort((x, y) => (excluded.has(x.a) || excluded.has(x.b)) - (excluded.has(y.a) || excluded.has(y.b))
        || y.match_score - x.match_score).map((p) => {
        const st = status(p);
        return el("tr", { class: st === "Predicted" ? "rt-chosen" : excluded.has(p.a) || excluded.has(p.b) ? "rt-muted" : "" },
          el("td", {}, `${p.a} & ${p.b}`),
          el("td", { class: "num strong" }, p.match_score),
          ...r.factors.map((f) => el("td", { class: `num${r.weights[f.key] ? "" : " off"}` }, pct(p.components[f.key]))),
          el("td", {}, st));
      }));
  }

  // ---- render -------------------------------------------------------------
  function render(r) {
    const n = r.face_count;
    const p = r.prediction;
    els.count.replaceChildren();
    if (n === 0) els.count.append("No faces found.");
    else {
      els.count.append(el("strong", {}, `${n} ${n === 1 ? "face" : "faces"}`), " found");
      if (r.photo_count > 1) els.count.append(` in ${r.photo_count} photos`);
      if (r.people_count != null && r.people_count < n) {
        els.count.append(" (", el("strong", {}, `${r.people_count} different people`),
          ": the same person in more than one photo is counted once)");
      }
      if (n >= 2) els.count.append(", ", el("strong", {}, `${r.pair_count} pairs`), " rated");
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
          breakdown(x.breakdown, r.faces.some((f) => f.features.attractiveness) ? [] : ["attractiveness"])));
    });
    if (!cards.length) {
      cards.push(el("article", { class: "match-card none" },
        el("div", { class: "match-text" }, el("p", { class: "match-names" }, "No pair predicted"))));
    }
    els.matchList.replaceChildren(...cards);
    const notes = [p.message, "Every rating is listed under “All ratings”."];
    if (p.pairs.length) notes.push("This is an experimental prediction, not proof that anyone is a couple.");
    els.matchNote.textContent = notes.join(" ");

    // photos
    els.annotatedList.replaceChildren(...r.annotated_images.map((src, i) => el("figure", { class: "annotated" },
      el("img", { src, alt: `Photo ${i + 1} with detected faces outlined${p.pairs.length ? "; predicted pairs highlighted" : ""}.` }),
      el("figcaption", {}, (r.photo_count > 1 ? `Photo ${i + 1}. ` : "") + (p.pairs.length
        ? "Predicted pairs are highlighted in pink; a line joins a pair in the same photo."
        : n ? "Boxes show where faces were found. People are labelled from left to right."
          : "No faces found. Faces may be too small, turned away, covered or poorly lit.")))));

    // gallery: every person's result, so nobody is left unexplained
    const status = Object.fromEntries((p.people_status || []).map((s) => [s.label, s]));
    els.gallery.replaceChildren(...r.crops.map((c) => {
      const st = status[c.label] || { status: "", detail: "" };
      return el("li", { class: `${c.in_pair ? "in-pair" : ""} st-${st.status}`, style: `--c:${c.colour}` },
        el("img", { src: c.image, alt: `Face crop of ${c.label}` }),
        el("div", { class: "who" }, el("span", { class: "swatch" }), c.label + (r.photo_count > 1 ? ` · photo ${c.photo}` : "")),
        el("div", { class: "meta" }, st.detail));
    }));
    els.galleryBlock.hidden = n === 0;

    if (r.factors && !weights) {
      factors = r.factors;
      weights = { ...r.weights };
    }
    renderWeights();
    els.weightsPanel.hidden = n < 2;
    els.ratingsPanel.hidden = n === 0;
    if (n) renderRatings(r);
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
    files = [];
    answers = null;
    lastResult = null;
    els.form.reset();
    previewUrls.forEach((u) => URL.revokeObjectURL(u));
    previewUrls = [];
    els.strip.replaceChildren();
    els.strip.hidden = true;
    els.dropEmpty.hidden = false;
    els.fileName.hidden = true;
    els.analyse.disabled = true;
    els.analyse.textContent = "Analyse Photo";
    els.change.hidden = true;
    els.results.hidden = true;
    els.annotatedList.replaceChildren();
    showError("");
    setStatus("");
    window.scrollTo({ top: 0 });
    els.input.focus();
  });
})();
