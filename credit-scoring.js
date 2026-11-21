
const tabs = document.querySelectorAll(".mc-card");
const panels = document.querySelectorAll(".mc-panel");

function openTab(name, scroll) {
    tabs.forEach(tab => {
        tab.classList.toggle("active", tab.dataset.tab === name);
    });
    panels.forEach(panel => {
        panel.classList.toggle("active", panel.id === name);
    });
    if (scroll) {
        document.getElementById(name).scrollIntoView({ behavior: "smooth", block: "start" });
    }
}

tabs.forEach(tab => {
    tab.addEventListener("click", event => {
        event.preventDefault();
        openTab(tab.dataset.tab, true);
        history.replaceState(null, "", "#" + tab.dataset.tab);
    });
});

const fromHash = window.location.hash.replace("#", "");
const hashPanel = document.getElementById(fromHash);
if (hashPanel && hashPanel.classList.contains("mc-panel")) {
    openTab(fromHash, false);
}


/* ---------- TAB 2: kode dan hasil ---------- */
const KODE_URL = "../1_Model/credit_scoring.py";
const HASIL_URL = "../1_Model/outputs/hasil.json";

const codeText = document.getElementById("codeText");
const copyButton = document.getElementById("copyCode");

fetch(KODE_URL)
    .then(response => {
        if (!response.ok) throw new Error("not found");
        return response.text();
    })
    .then(text => {
        codeText.textContent = text.trimEnd();
    })
    .catch(() => {
        codeText.textContent =
            "Could not load credit_scoring.py. Open this page with Live Server " +
            "or the published site, not by double-clicking the file.";
    });

copyButton.addEventListener("click", () => {
    navigator.clipboard.writeText(codeText.textContent).then(() => {
        copyButton.textContent = "Copied";
        setTimeout(() => {
            copyButton.textContent = "Copy Code";
        }, 1500);
    });
});

const formatPersen = x => (x * 100).toFixed(1) + "%";
const formatRupiah = x => "Rp" + Math.round(x).toLocaleString("en-US");

fetch(HASIL_URL)
    .then(response => {
        if (!response.ok) throw new Error("not found");
        return response.json();
    })
    .then(h => {
        const setText = (id, text) => { document.getElementById(id).textContent = text; };
        const nf = x => x.toLocaleString("en-US");

        setText("mMeta", "Logistic Regression vs Random Forest, evaluated on data the models never saw during training.");
        setText("mN", nf(h.jumlah_nasabah));
        setText("mDefault", formatPersen(h.tingkat_gagal_bayar));
        setText("mTrain", nf(h.jumlah_training) + " / " + nf(h.jumlah_test));
        setText("mThreshold", h.ambang.toFixed(2));

        const metricRows = [["Accuracy", "akurasi"], ["Precision", "presisi"], ["Recall", "recall"],
                            ["F1 score", "f1"], ["AUC", "AUC"], ["Gini", "Gini"], ["KS", "KS"]];
        document.getElementById("modelRows").innerHTML = metricRows.map(([label, key]) =>
            "<tr><td>" + label + "</td><td>" + h.metrik.lr[key].toFixed(3) +
            "</td><td>" + h.metrik.rf[key].toFixed(3) + "</td></tr>").join("");

        const names = { Rendah: "Low", Sedang: "Medium", Tinggi: "High" };
        document.getElementById("riskRows").innerHTML = ["Rendah", "Sedang", "Tinggi"]
            .filter(k => h.risk_level[k])
            .map(k => "<tr><td>" + names[k] + "</td><td>" + nf(h.risk_level[k].jumlah) +
                      "</td><td>" + formatPersen(h.risk_level[k].gagal_bayar) + "</td></tr>").join("");
    })
    .catch(() => {
        document.getElementById("mMeta").textContent =
            "Results file not found yet. Run credit_scoring.py once to create hasil.json.";
    });

 
/* ---------- TAB 3: BORROWER SCORER ---------- */
const MODEL_URL = "../1_Model/outputs/model.json";
const scoreResults = document.getElementById("simResults");
const fmtInt = x => x.toLocaleString("en-US");

let model = null, groupStats = null, nBorrowers = 0, baseRate = 0, ambang = 0.5, hasilData = null;

const LEVELS = [
    { key: "Rendah", label: "Low", cls: "lvl-low" },
    { key: "Sedang", label: "Medium", cls: "lvl-mid" },
    { key: "Tinggi", label: "High", cls: "lvl-high" }
];

function readInputs() {
    const vals = {}, cats = {}, outside = [];
    model.angka.forEach(f => {
        const el = document.querySelector('[data-f="' + f.nama + '"]');
        let x = parseFloat(el.value);
        if (isNaN(x)) x = f.median;          // empty = typical value, like the imputer in training
        if (x < f.min || x > f.maks) outside.push(f.nama);
        vals[f.nama] = x;
    });
    model.kategori.forEach(f => {
        cats[f.nama] = document.querySelector('[data-c="' + f.nama + '"]').value;
    });
    return { vals, cats, outside };
}

const FEATURE_LABELS = {
    person_age: "Age", person_income: "Annual income", person_emp_length: "Employment length",
    loan_amnt: "Loan amount", loan_int_rate: "Interest rate",
    cb_person_cred_hist_length: "Credit history length",
    cb_person_default_on_file_code: "Previous default", person_home_ownership: "Home ownership"
};

// same formula as the Logistic Regression trained in credit_scoring.py
function scoreBorrower(vals, cats) {
    const sk = model.skor;
    let z = model.intercept;
    const parts = [];
    model.angka.forEach(f => {
        const c = f.koefisien * (vals[f.nama] - f.mean) / f.scale;
        z += c;
        parts.push({ name: f.nama, c });          // already relative to the average borrower
    });
    model.kategori.forEach(f => {
        const coefOf = k => (k === f.dasar ? 0 : (f.koefisien[k] || 0));
        const c = coefOf(cats[f.nama]);
        z += c;
        const typical = f.kategori.reduce((s, k) => s + (f.frekuensi ? f.frekuensi[k] : 0) * coefOf(k), 0);
        parts.push({ name: f.nama, c: c - typical });
    });
    const p = 1 / (1 + Math.exp(-z));
    const s = sk.minimal + (1 - p) * (sk.maksimal - sk.minimal);
    const pSafe = Math.min(Math.max(p, sk.batas_pd), 1 - sk.batas_pd);
    const odds = (1 - pSafe) / pSafe;
    const scorecard = sk.dasar + (sk.pdo / Math.log(2)) * Math.log(odds / sk.odds_dasar);
    const level = s >= sk.batas_rendah ? LEVELS[0] : s >= sk.batas_sedang ? LEVELS[1] : LEVELS[2];
    return { p, s, scorecard, odds, level, parts };
}

function scaleSvg(score, sk) {
    const W = 700, H = 100, L = 20, R = 20, y = 44, h = 22;
    const X = v => L + (v - sk.minimal) / (sk.maksimal - sk.minimal) * (W - L - R);
    const zone = (a, b, col, name) =>
        '<rect x="' + X(a).toFixed(1) + '" y="' + y + '" width="' + (X(b) - X(a)).toFixed(1) + '" height="' + h +
        '" fill="' + col + '"/><text x="' + ((X(a) + X(b)) / 2).toFixed(1) + '" y="' + (y + 15) +
        '" font-size="11" fill="#3a1e20" text-anchor="middle">' + name + "</text>";
    const tick = v => '<text x="' + X(v).toFixed(1) + '" y="' + (y + h + 16) +
        '" font-size="11" fill="#806d68" text-anchor="middle">' + v + "</text>";
    const sc = Math.max(sk.minimal, Math.min(sk.maksimal, score));
    const frac = (sc - sk.minimal) / (sk.maksimal - sk.minimal);
    const anchor = frac > 0.85 ? "end" : frac < 0.15 ? "start" : "middle";
    const tx = X(sc) + (anchor === "end" ? 4 : anchor === "start" ? -4 : 0);
    return '<svg viewBox="0 0 ' + W + " " + H + '">' +
        zone(sk.minimal, sk.batas_sedang, "#e3b5b0", "High") +
        zone(sk.batas_sedang, sk.batas_rendah, "#efd9a8", "Medium") +
        zone(sk.batas_rendah, sk.maksimal, "#b9d8bf", "Low") +
        '<line x1="' + X(sc).toFixed(1) + '" x2="' + X(sc).toFixed(1) + '" y1="' + (y - 12) + '" y2="' + (y + h + 4) +
        '" stroke="#682127" stroke-width="3"/>' +
        '<text x="' + tx.toFixed(1) + '" y="' + (y - 18) + '" font-size="12" fill="#682127" text-anchor="' + anchor +
        '">Score ' + Math.round(score) + "</text>" +
        tick(sk.minimal) + tick(sk.batas_sedang) + tick(sk.batas_rendah) + tick(sk.maksimal) + "</svg>";
}

function factorsHtml(parts) {
    const rows = parts.map(x => ({ label: FEATURE_LABELS[x.name] || x.name, c: x.c }))
        .sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
    const max = Math.max(...rows.map(r => Math.abs(r.c)), 0.0001);
    return rows.map(r => {
        const w = (Math.abs(r.c) / max * 100).toFixed(0);
        return '<div class="mc-factor"><span>' + r.label + "</span>" +
            '<div class="half l">' + (r.c < 0 ? '<i class="down" style="width:' + w + '%"></i>' : "") + "</div>" +
            '<div class="half">' + (r.c >= 0 ? '<i class="up" style="width:' + w + '%"></i>' : "") + "</div></div>";
    }).join("");
}

function renderScore() {
    const { vals, cats, outside } = readInputs();
    const r = scoreBorrower(vals, cats);
    const sk = model.skor;
    const g = groupStats[r.level.key];
    const oddsText = r.odds >= 100 ? Math.round(r.odds) : r.odds.toFixed(1);

    scoreResults.innerHTML =
        '<div class="mc-res-head"><h3>Estimated risk for this borrower</h3>' +
        "<span>Logistic Regression scorecard · trained on " + fmtInt(nBorrowers) + " borrowers</span></div>" +

        '<div class="mc-metrics">' +
        '<article class="mc-metric"><span>Probability of default</span><strong>' + formatPersen(r.p) +
        "</strong><small>chance this borrower defaults</small></article>" +
        '<article class="mc-metric"><span>Credit score</span><strong>' + Math.round(r.s) +
        "</strong><small>on a " + sk.minimal + "–" + sk.maksimal + " scale</small></article>" +
        '<article class="mc-metric"><span>Risk level</span><strong class="' + r.level.cls + '">' + r.level.label +
        "</strong><small>" + formatPersen(g.gagal_bayar) + " of " + fmtInt(g.jumlah) +
        " past borrowers in this group defaulted</small></article>" +
        '<article class="mc-metric"><span>Scorecard score</span><strong>' + Math.round(r.scorecard) +
        "</strong><small>odds " + oddsText + " good : 1 bad</small></article></div>" +

        '<div class="mc-chart-card"><h3>Where this score sits</h3>' +
        "<p>The 300–850 scale is split into three risk levels. The marker shows this borrower. " +
        "The scorecard score uses " + sk.dasar + " points at " + sk.odds_dasar + ":1 odds, plus " + sk.pdo +
        " points every time the odds double.</p>" + scaleSvg(r.s, sk) + "</div>" +

        '<div class="mc-chart-card"><h3>What drives this result</h3>' +
        "<p>How much each field moves the risk compared with a typical borrower in the training data.</p>" +
        '<div class="mc-factors-head"><span></span><span>lowers risk</span><span>raises risk</span></div>' +
        factorsHtml(r.parts) + "</div>" +

        (outside.length
            ? '<p class="mc-hint">⚠ Some values are outside the range seen in the training data, so this estimate is less reliable.</p>'
            : "");
}


Promise.all([
    fetch(MODEL_URL).then(r => { if (!r.ok) throw new Error("model"); return r.json(); }),
    fetch(HASIL_URL).then(r => { if (!r.ok) throw new Error("hasil"); return r.json(); })
]).then(([m, h]) => {
    model = m;
    groupStats = h.risk_level;
    nBorrowers = h.jumlah_nasabah;
    baseRate = h.tingkat_gagal_bayar;
    ambang = h.ambang;
    hasilData = h;
    renderScore();
    const panel = document.querySelector(".mc-input-panel");
    panel.addEventListener("input", renderScore);
    panel.addEventListener("change", renderScore);
}).catch(() => {
    scoreResults.innerHTML =
        '<p class="mc-placeholder">Model file not found yet. Run credit_scoring.py once to create model.json and hasil.json.</p>';
});


/* ---------- TAB 3: BATCH SCORING (CSV) ---------- */
const BATCH_MAX_ROWS = 5000;
let batchRows = [];   // kept for the Excel download (next step)

const escHtml = s => String(s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function csvSplit(line, sep) {
    const out = [];
    let cur = "", quoted = false;
    for (const ch of line) {
        if (ch === '"') quoted = !quoted;
        else if (ch === sep && !quoted) { out.push(cur.trim()); cur = ""; }
        else cur += ch;
    }
    out.push(cur.trim());
    return out;
}

function csvNumber(raw, sep) {
    let s = String(raw).replace(/[^\d.,-]/g, "");
    if (!s) return NaN;
    s = sep === ";" ? s.replace(/\./g, "").replace(",", ".") : s.replace(/,/g, "");
    return parseFloat(s);
}

function downloadFile(name, content, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

function borrowerColumns() {
    return model.angka.map(f => f.nama).concat(model.kategori.map(f => f.nama));
}

const batchStatus = document.getElementById("batchStatus");
const batchOut = document.getElementById("batchOut");

/* ---------- BATCH: FLEXIBLE COLUMN MATCHING ---------- */
const mapBox = document.getElementById("mapBox");
let batchFileData = null, batchMap = {}, batchIdCol = -1;

const FIELD_INFO = {
    person_age: { label: "Age", alias: ["age", "umur", "usia"] },
    person_income: { label: "Annual income", alias: ["income", "salary", "gaji", "penghasilan", "pendapatan"] },
    person_emp_length: { label: "Employment length (years)", alias: ["emp_length", "employment", "masa_kerja", "lama_kerja", "tenure", "years_employed"] },
    loan_amnt: { label: "Loan amount", alias: ["loan_amount", "loan_amnt", "amount", "pinjaman", "plafon"] },
    loan_int_rate: { label: "Interest rate (%)", alias: ["int_rate", "interest", "bunga", "rate"] },
    cb_person_cred_hist_length: { label: "Credit history length (years)", alias: ["cred_hist", "credit_history", "history", "riwayat", "lama_kredit"] },
    cb_person_default_on_file_code: { label: "Previous default (yes/no)", alias: ["default_on_file", "previous_default", "default", "gagal_bayar", "tunggakan", "macet"] },
    person_home_ownership: { label: "Home ownership", alias: ["home_ownership", "ownership", "home", "rumah", "tempat_tinggal"] }
};

const norm = h => String(h).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
const YES = ["1", "y", "yes", "ya", "true", "t", "pernah"];
const NO = ["0", "n", "no", "tidak", "tdk", "false", "f", "belum"];

function normHome(raw) {
    const t = raw.toLowerCase().trim();
    if (!t) return null;
    if (/rent|sewa|kontrak|kos|kost/.test(t)) return "RENT";
    if (/mortgage|kpr|cicil/.test(t)) return "MORTGAGE";
    if (/own|milik|pribadi|lunas/.test(t)) return "OWN";
    if (/other|lain/.test(t)) return "OTHER";
    return null;
}

function modelFields() {
    return Object.keys(FIELD_INFO).filter(f => borrowerColumns().includes(f));
}

// guess which column holds which field: exact names first, then partial matches
function autoMap(heads) {
    const keys = heads.map(norm), map = {}, used = new Set(), names = modelFields();
    names.forEach(f => {
        const i = keys.findIndex((k, idx) => !used.has(idx) && (k === f || FIELD_INFO[f].alias.includes(k)));
        if (i >= 0) { map[f] = i; used.add(i); }
    });
    names.forEach(f => {
        if (map[f] !== undefined) return;
        let best = -1, bestLen = 0;
        keys.forEach((k, idx) => {
            if (used.has(idx)) return;
            FIELD_INFO[f].alias.forEach(a => {
                if (a.length >= 3 && k.includes(a) && a.length > bestLen) { best = idx; bestLen = a.length; }
            });
        });
        if (best >= 0) { map[f] = best; used.add(best); }
    });
    names.forEach(f => { if (map[f] === undefined) map[f] = -1; });
    return map;
}

function parseBatchFile(text) {
    const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(l => l.trim());
    if (lines.length < 2) throw new Error("the file has no data rows");
    if (lines.length - 1 > BATCH_MAX_ROWS) throw new Error("more than " + fmtInt(BATCH_MAX_ROWS) + " rows");
    const c = (lines[0].match(/,/g) || []).length;
    const s = (lines[0].match(/;/g) || []).length;
    const t = (lines[0].match(/\t/g) || []).length;
    const sep = t > c && t > s ? "\t" : s > c ? ";" : ",";
    return { sep, head: csvSplit(lines[0], sep), rows: lines.slice(1).map(l => csvSplit(l, sep)) };
}

function renderMapping() {
    const opts = (sel, none) =>
        '<option value="-1"' + (sel === -1 ? " selected" : "") + ">" + none + "</option>" +
        batchFileData.head.map((h, i) =>
            '<option value="' + i + '"' + (sel === i ? " selected" : "") + ">" +
            escHtml(h || "(column " + (i + 1) + ")") + "</option>").join("");

    mapBox.innerHTML =
        '<div class="mc-subheading"><span>COLUMN MATCHING</span><h3>Check how your columns match</h3></div>' +
        '<p class="mc-caption">We matched your columns automatically. Change any dropdown if a match is wrong; the results update right away.</p>' +
        '<div class="mc-map">' +
        modelFields().map(f =>
            '<div class="mc-map-row"><label>' + FIELD_INFO[f].label + '</label><select data-map="' + f + '">' +
            opts(batchMap[f], "— not in my file (typical value) —") + "</select></div>").join("") +
        '<div class="mc-map-row"><label>Borrower ID or name (optional)</label><select data-map="__id">' +
        opts(batchIdCol, "— none, number the rows —") + "</select></div></div>";
}

mapBox.addEventListener("change", e => {
    const f = e.target.dataset.map;
    if (!f) return;
    const v = parseInt(e.target.value, 10);
    if (f === "__id") batchIdCol = v; else batchMap[f] = v;
    scoreBatch();
});

function scoreBatch() {
    const fields = modelFields();
    const matched = fields.filter(f => batchMap[f] >= 0).length;
    if (matched < 4) {
        batchOut.innerHTML = "";
        excelBtn.hidden = true;
        batchStatus.textContent = "Only " + matched + " of " + fields.length +
            " fields are matched. Match at least 4 so the results mean something.";
        return;
    }
    const D = batchFileData;

    // interest rates written as fractions (0.11) are converted to percent (11)
    let rateScale = 1;
    const ri = batchMap.loan_int_rate;
    if (ri >= 0) {
        const vs = D.rows.map(c => csvNumber(c[ri] || "", D.sep)).filter(v => !isNaN(v));
        if (vs.length && Math.max(...vs) <= 1) rateScale = 100;
    }

    let replaced = 0, outsideRows = 0;
    const rows = D.rows.map((cells, n) => {
        const vals = {}, cats = {};
        let outside = false;

        model.angka.forEach(f => {
            const ci = batchMap[f.nama] === undefined ? -1 : batchMap[f.nama];
            const raw = ci >= 0 ? (cells[ci] || "").trim() : "";
            let x;
            if (f.biner) {
                const t = raw.toLowerCase();
                x = YES.includes(t) ? 1 : NO.includes(t) ? 0 : NaN;
            } else {
                x = csvNumber(raw, D.sep);
                if (f.nama === "loan_int_rate") x *= rateScale;
            }
            if (isNaN(x)) { x = f.median; if (ci >= 0) replaced++; }
            else if (x < f.min || x > f.maks) outside = true;
            vals[f.nama] = x;
        });

        model.kategori.forEach(f => {
            const ci = batchMap[f.nama] === undefined ? -1 : batchMap[f.nama];
            const raw = ci >= 0 ? (cells[ci] || "").trim() : "";
            let c = f.kategori.includes(raw.toUpperCase()) ? raw.toUpperCase()
                  : (f.nama === "person_home_ownership" ? normHome(raw) : null);
            if (c && !f.kategori.includes(c)) c = null;
            if (!c) { c = f.modus; if (ci >= 0) replaced++; }
            cats[f.nama] = c;
        });
        if (outside) outsideRows++;

        const r = scoreBorrower(vals, cats);
        const q = rf ? scoreRF(vals, cats) : null;
        return {
            no: batchIdCol >= 0 && cells[batchIdCol] ? cells[batchIdCol] : String(n + 1),
            vals, cats, p: r.p, s: r.s, scorecard: r.scorecard, odds: r.odds,
            level: r.level, outside, parts: r.parts, raw: cells,
            pRf: q ? q.p : null, sRf: q ? q.s : null, levelRf: q ? q.level : null
        };
    });

    batchRows = rows;
    excelBtn.hidden = false;
    batchStatus.textContent = fmtInt(rows.length) + " borrowers scored · " + matched + " of " + fields.length +
        " fields matched" + (rateScale === 100 ? " · interest rates looked like fractions, so they were multiplied by 100" : "") + ".";
    renderBatch({
        rows,
        missing: fields.filter(f => batchMap[f] < 0).map(f => FIELD_INFO[f].label),
        replaced,
        outsideRows
    });
}

async function handleBatchFile(file) {
    if (!file) return;
    if (!model) { batchStatus.textContent = "The model is still loading. Try again in a moment."; return; }
    if (file.size > 5 * 1024 * 1024) { batchStatus.textContent = "The file is larger than 5 MB."; return; }
    try {
        batchFileData = parseBatchFile(await file.text());
        batchMap = autoMap(batchFileData.head);
        batchIdCol = batchFileData.head.findIndex(h =>
            ["id", "no", "name", "nama", "borrower", "customer", "nasabah"].includes(norm(h)));
        renderMapping();
        scoreBatch();
    } catch (err) {
        batchRows = [];
        excelBtn.hidden = true;
        batchOut.innerHTML = "";
        mapBox.innerHTML = "";
        batchStatus.textContent = "Could not read the file: " + err.message;
    }
}


const batchDrop = document.getElementById("batchDrop");
const batchFile = document.getElementById("batchFile");
batchFile.addEventListener("change", e => { handleBatchFile(e.target.files[0]); batchFile.value = ""; });
["dragenter", "dragover"].forEach(ev =>
    batchDrop.addEventListener(ev, e => { e.preventDefault(); batchDrop.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
    batchDrop.addEventListener(ev, e => { e.preventDefault(); batchDrop.classList.remove("dragover"); }));
batchDrop.addEventListener("drop", e => handleBatchFile(e.dataTransfer.files[0]));


/* ---------- RANDOM FOREST IN THE BROWSER ---------- */
const RF_URL = "../1_Model/outputs/rf.json";
let rf = null;

// same preprocessing as the Pipeline: scale numbers, one-hot categories (first category dropped)
function featureVector(vals, cats) {
    return rf.nama_fitur.map(name => {
        const num = model.angka.find(f => f.nama === name);
        if (num) return (vals[num.nama] - num.mean) / num.scale;
        for (const f of model.kategori) {
            if (name.startsWith(f.nama + "_")) return cats[f.nama] === name.slice(f.nama.length + 1) ? 1 : 0;
        }
        return 0;
    });
}

function scoreRF(vals, cats) {
    const x = featureVector(vals, cats);
    let sum = 0;
    rf.pohon.forEach(t => {
        let i = 0;
        while (t.kiri[i] !== -1) i = x[t.fitur[i]] <= t.batas[i] ? t.kiri[i] : t.kanan[i];
        sum += t.prob[i];
    });
    const p = sum / rf.pohon.length;
    const sk = model.skor;
    const s = sk.minimal + (1 - p) * (sk.maksimal - sk.minimal);
    const level = s >= sk.batas_rendah ? LEVELS[0] : s >= sk.batas_sedang ? LEVELS[1] : LEVELS[2];
    return { p, s, level };
}

fetch(RF_URL)
    .then(r => { if (!r.ok) throw new Error("rf"); return r.json(); })
    .then(j => { rf = j; })
    .catch(() => { rf = null; });



/* ---------- BATCH RESULTS (LR + RF) ---------- */
function batchHist(values, lo, hi, bins, colorFor, marks, fmt) {
    const W = 460, H = 210, L = 12, R = 12, B = 36, base = H - B;
    const w = (hi - lo) / bins, counts = new Array(bins).fill(0);
    values.forEach(v => { counts[Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / w)))]++; });
    const top = Math.max(...counts, 1), bw = (W - L - R) / bins;
    const X = v => L + (v - lo) / (hi - lo) * (W - L - R);
    let out = '<svg viewBox="0 0 ' + W + " " + H + '">';
    counts.forEach((c, i) => {
        const h = c / top * (base - 30);
        out += '<rect x="' + (L + i * bw + 0.5).toFixed(1) + '" y="' + (base - h).toFixed(1) + '" width="' +
               (bw - 1).toFixed(1) + '" height="' + h.toFixed(1) + '" fill="' + colorFor(lo + (i + 0.5) * w) + '"/>';
    });
    out += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + base + '" y2="' + base + '" stroke="#806d68"/>';
    marks.forEach(m => {
        const x = X(m.v).toFixed(1);
        out += '<line x1="' + x + '" x2="' + x + '" y1="20" y2="' + base +
               '" stroke="#682127" stroke-width="2" stroke-dasharray="4 3"/>' +
               '<text x="' + x + '" y="14" font-size="11" fill="#682127" text-anchor="middle">' + m.label + "</text>";
    });
    for (let k = 0; k <= 4; k++) {
        const v = lo + (hi - lo) * k / 4;
        out += '<text x="' + X(v).toFixed(1) + '" y="' + (H - 14) + '" font-size="11" fill="#806d68" text-anchor="' +
               (k === 0 ? "start" : k === 4 ? "end" : "middle") + '">' + fmt(v) + "</text>";
    }
    return out + "</svg>";
}

function renderBatch(res) {
    const rows = res.rows, n = rows.length, sk = model.skor;
    const avgPd = rows.reduce((a, r) => a + r.p, 0) / n;
    const avgScore = rows.reduce((a, r) => a + r.s, 0) / n;
    const expected = rows.reduce((a, r) => a + r.p, 0);
    const predicted = rows.filter(r => r.p >= ambang).length;

    const groups = LEVELS.map(lv => {
        const g = rows.filter(r => r.level.key === lv.key);
        return { lv, n: g.length, share: g.length / n,
                 avgPd: g.length ? g.reduce((a, r) => a + r.p, 0) / g.length : 0,
                 hist: groupStats[lv.key] };
    });
    const [gLow, gMid, gHigh] = groups;

    // average push of each field, relative to a typical training borrower
    const drv = {};
    rows.forEach(r => r.parts.forEach(x => { drv[x.name] = (drv[x.name] || 0) + x.c / n; }));
    const list = Object.keys(drv).map(k => ({ label: FEATURE_LABELS[k] || k, c: drv[k] })).sort((a, b) => b.c - a.c);
    const raising = list.filter(d => d.c > 0.02).slice(0, 3).map(d => d.label);
    const lowering = list.slice().reverse().filter(d => d.c < -0.02).slice(0, 3).map(d => d.label);

    const diff = avgPd - baseRate;
    const vsBase = Math.abs(diff) < 0.01 ? "about the same as" : (diff > 0 ? "higher than" : "lower than");

    const notes = [];
    if (res.missing.length) notes.push("Columns not found, typical values used: " + res.missing.join(", ") + ".");
    if (res.replaced) notes.push(fmtInt(res.replaced) + " empty or unreadable values were replaced with typical training values.");
    if (res.outsideRows) notes.push(fmtInt(res.outsideRows) + " rows have values outside the training range, so their estimates are less reliable.");

    const rowHtml = r =>
        "<tr><td>" + escHtml(r.no) + "</td><td>" + r.vals.person_age + "</td><td>" + fmtInt(r.vals.person_income) +
        "</td><td>" + fmtInt(r.vals.loan_amnt) + "</td><td>" + r.vals.loan_int_rate + "</td><td>" + formatPersen(r.p) +
        "</td><td>" + Math.round(r.s) + '</td><td><span class="lvl-chip ' + r.level.cls + '">' + r.level.label + "</span></td>" +
        (r.pRf === null ? "<td>–</td><td>–</td><td>–</td>"
            : "<td>" + formatPersen(r.pRf) + "</td><td>" + Math.round(r.sRf) +
              '</td><td><span class="lvl-chip ' + r.levelRf.cls + '">' + r.levelRf.label + "</span></td>") +
        "</tr>";
    const top10 = rows.slice().sort((a, b) => b.p - a.p).slice(0, 10).map(rowHtml).join("");

    const groupRows = groups.map(g =>
        '<tr><td><span class="lvl-chip ' + g.lv.cls + '">' + g.lv.label + "</span></td><td>" + fmtInt(g.n) +
        '</td><td><span class="mc-share"><i style="width:' + (g.share * 100).toFixed(0) + '%"></i></span>' + formatPersen(g.share) +
        "</td><td>" + (g.n ? formatPersen(g.avgPd) : "–") + "</td><td>" +
        (g.hist ? formatPersen(g.hist.gagal_bayar) : "–") + "</td></tr>").join("");

    const pdSvg = batchHist(rows.map(r => r.p), 0, 1, 20,
        c => (c >= ambang ? "#c98a85" : "#682127"),
        [{ v: ambang, label: "threshold " + ambang.toFixed(2) }], v => v.toFixed(1));
    const scSvg = batchHist(rows.map(r => r.s), sk.minimal, sk.maksimal, 28,
        c => (c >= sk.batas_rendah ? "#7fb08a" : c >= sk.batas_sedang ? "#e0b95a" : "#c98a85"),
        [{ v: sk.batas_sedang, label: sk.batas_sedang }, { v: sk.batas_rendah, label: sk.batas_rendah }],
        v => Math.round(v));

    let adviceHigh;
    if (gHigh.share >= 0.3) {
        adviceHigh = "<strong>A large high-risk group.</strong> Consider tighter approval criteria, collateral, or a smaller loan amount for these borrowers.";
    } else if (gHigh.share >= 0.1) {
        adviceHigh = "<strong>A meaningful high-risk group.</strong> Send these borrowers to manual review before approving.";
    } else {
        adviceHigh = "<strong>Few high-risk borrowers.</strong> Standard checks should be enough, with a closer look at the highest-risk cases listed below.";
    }

    const warning = gHigh.n
        ? fmtInt(gHigh.n) + " borrowers (" + formatPersen(gHigh.share) + " of the file) are High risk. In the training data, " +
          (gHigh.hist ? formatPersen(gHigh.hist.gagal_bayar) : "a large share") + " of borrowers in this group actually defaulted, " +
          "compared with " + formatPersen(baseRate) + " overall. A probability is an average, so individual borrowers can still go either way."
        : "No borrower in this file falls into the High risk level. Scores still depend on the model and on the data you uploaded.";

    batchOut.innerHTML =
        '<div class="mc-res-head"><h3>Portfolio results</h3><span>' + fmtInt(n) +
        " borrowers · Logistic Regression scorecard · trained on " + fmtInt(nBorrowers) + " borrowers</span></div>" +

        '<div class="mc-metrics">' +
        '<article class="mc-metric"><span>Borrowers scored</span><strong>' + fmtInt(n) + "</strong><small>rows in your file</small></article>" +
        '<article class="mc-metric"><span>Average default probability</span><strong>' + formatPersen(avgPd) +
        "</strong><small>" + vsBase + " the training data (" + formatPersen(baseRate) + ")</small></article>" +
        '<article class="mc-metric"><span>Average credit score</span><strong>' + Math.round(avgScore) +
        "</strong><small>on a " + sk.minimal + "–" + sk.maksimal + " scale</small></article>" +
        '<article class="mc-metric"><span>Predicted to default</span><strong>' + fmtInt(predicted) +
        "</strong><small>PD ≥ " + ambang.toFixed(2) + " · about " + fmtInt(Math.round(expected)) + " expected from summed PD</small></article></div>" +

        (notes.length ? '<p class="mc-hint">⚠ ' + notes.join(" ") + "</p>" : "") +

        '<div class="mc-chart-card"><h3>Risk levels in this file</h3>' +
        '<p class="mc-caption">Share of borrowers per level, next to the real default rate of each level in the training data.</p>' +
        '<div class="mc-table-wrap"><table class="mc-result-table"><thead><tr><th>RISK LEVEL</th><th>BORROWERS</th><th>SHARE</th>' +
        "<th>AVG PD IN FILE</th><th>TRAINING DEFAULT RATE</th></tr></thead><tbody>" + groupRows + "</tbody></table></div></div>" +

        '<div class="mc-res-two">' +
        '<div class="mc-chart-card"><h3>Probability of default</h3>' +
        '<p class="mc-caption">Distribution of PD across your borrowers. Darker bars are below the classification threshold.</p>' + pdSvg + "</div>" +
        '<div class="mc-chart-card"><h3>Credit score</h3>' +
        '<p class="mc-caption">Distribution of scores. Dashed lines mark the Medium and Low risk boundaries.</p>' + scSvg + "</div></div>" +

        '<div class="mc-warning"><h4>⚠ Watch out for the High risk group</h4><p>' + warning + "</p></div>" +

        '<div class="mc-res-two">' +
        '<div class="mc-read"><h4>How to read the results</h4>' +
        "<p><strong>PD, score and level.</strong> PD is the model's estimated chance of default. The score turns it into a " +
        sk.minimal + "–" + sk.maksimal + " scale, and the risk level groups scores into Low, Medium and High.</p>" +
        "<p><strong>Against the training data.</strong> Your file's average PD is " + formatPersen(avgPd) + ", " + vsBase +
        " the " + formatPersen(baseRate) + " default rate of the training borrowers.</p>" +
        "<p><strong>What pushes risk.</strong> " +
        (raising.length ? "On average, " + raising.join(", ") + " raise risk the most compared with a typical training borrower. " : "No single field clearly raises risk in this file. ") +
        (lowering.length ? lowering.join(", ") + " lower it." : "") + "</p></div>" +
        '<div class="mc-read"><h4>What you should do</h4>' +
        "<p>" + adviceHigh + "</p>" +
        (gMid.n ? "<p><strong>Medium risk.</strong> Approve with conditions, such as a shorter term or a smaller amount, and monitor repayments.</p>" : "") +
        (gLow.n ? "<p><strong>Low risk.</strong> Good candidates for faster approval, still subject to your normal checks.</p>" : "") +
        "</div></div>" +

        resultsCard(rows) +
        '<p class="mc-disclaimer">Scores come from models trained on ' + fmtInt(nBorrowers) +
        " borrowers. They are illustrative, depend on the data you upload, and are not a lending decision or a replacement for a bank's own credit policy.</p>";
}


/* ---------- LR vs RF COMPARISON ---------- */
const featLabel = n => FEATURE_LABELS[n] ||
    (n.indexOf("person_home_ownership_") === 0 ? "Home: " + n.slice(22) : n);

function levelCounts(rows, useRf) {
    const c = { Low: 0, Medium: 0, High: 0 };
    rows.forEach(r => { c[(useRf ? r.levelRf : r.level).label]++; });
    return c;
}

function modelCompareHtml(rows) {
    if (!rows.length || rows[0].pRf === null) {
        return '<p class="mc-hint">Random Forest results are not available yet. Run credit_scoring.py again so rf.json is created.</p>';
    }
    const n = rows.length;
    const avg = f => rows.reduce((a, r) => a + f(r), 0) / n;
    const cl = levelCounts(rows, false), cr = levelCounts(rows, true);
    const same = rows.filter(r => r.level.key === r.levelRf.key).length / n;
    const tr = (label, a, b) => "<tr><td>" + label + "</td><td>" + a + "</td><td>" + b + "</td></tr>";
    const m = hasilData && hasilData.metrik;

    return '<div class="mc-chart-card"><h3>Logistic Regression vs Random Forest</h3>' +
        '<p class="mc-caption">The same borrowers scored by both models. Logistic Regression is easy to explain; ' +
        "Random Forest can capture more complex patterns. The last three rows were measured on test data during training.</p>" +
        '<div class="mc-table-wrap"><table class="mc-result-table"><thead><tr><th>MEASURE</th><th>LOGISTIC REGRESSION</th>' +
        "<th>RANDOM FOREST</th></tr></thead><tbody>" +
        tr("Average default probability", formatPersen(avg(r => r.p)), formatPersen(avg(r => r.pRf))) +
        tr("Average credit score", Math.round(avg(r => r.s)), Math.round(avg(r => r.sRf))) +
        tr("Predicted to default (PD ≥ " + ambang.toFixed(2) + ")", fmtInt(rows.filter(r => r.p >= ambang).length),
           fmtInt(rows.filter(r => r.pRf >= ambang).length)) +
        tr("Low risk", fmtInt(cl.Low), fmtInt(cr.Low)) +
        tr("Medium risk", fmtInt(cl.Medium), fmtInt(cr.Medium)) +
        tr("High risk", fmtInt(cl.High), fmtInt(cr.High)) +
        (m ? tr("AUC (test data)", m.lr.AUC.toFixed(3), m.rf.AUC.toFixed(3)) +
             tr("Gini (test data)", m.lr.Gini.toFixed(3), m.rf.Gini.toFixed(3)) +
             tr("KS (test data)", m.lr.KS.toFixed(3), m.rf.KS.toFixed(3)) : "") +
        "</tbody></table></div>" +
        '<p class="mc-caption">The two models put ' + formatPersen(same) + " of your borrowers in the same risk level.</p></div>";
}

function barsHtml(items) {
    const max = Math.max(...items.map(i => i.v), 0.0001);
    return items.map(i =>
        '<div class="mc-hbar"><span>' + i.label + "</span><div><i style=\"width:" + (i.v / max * 100).toFixed(0) +
        '%"></i></div><b>' + i.text + "</b></div>").join("");
}

function modelReliesHtml() {
    if (!hasilData) return "";
    const imp = barsHtml(hasilData.importance.map(x =>
        ({ label: featLabel(x.fitur), v: x.importance, text: x.importance.toFixed(3) })));
    const odds = hasilData.koefisien.map(x =>
        "<tr><td>" + featLabel(x.fitur) + "</td><td>" + x.koefisien.toFixed(3) + "</td><td>" + x.rasio_odds.toFixed(2) + "</td></tr>").join("");

    return '<div class="mc-res-two">' +
        '<div class="mc-chart-card"><h3>What the Random Forest relies on</h3>' +
        '<p class="mc-caption">The ten fields the Random Forest uses most when judging default risk.</p>' + imp + "</div>" +
        '<div class="mc-chart-card"><h3>Logistic Regression odds ratios</h3>' +
        '<p class="mc-caption">An odds ratio above 1 raises the odds of default, below 1 lowers them. Numeric fields are standardised, so each ratio is per one standard deviation.</p>' +
        '<div class="mc-table-wrap"><table class="mc-result-table"><thead><tr><th>FIELD</th><th>COEFFICIENT</th><th>ODDS RATIO</th></tr></thead><tbody>' +
        odds + "</tbody></table></div></div></div>";
}



/* ---------- EXCEL DOWNLOAD ---------- */
const excelBtn = document.getElementById("excelBtn");
const LEVEL_FILL = { Rendah: "FFC6EFCE", Sedang: "FFFFEB9C", Tinggi: "FFFFC7CE" };

function styleHeader(ws) {
    ws.getRow(1).eachCell(c => {
        c.font = { bold: true, color: { argb: "FFFFFFFF" } };
        c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "FF1F4E78" } };
    });
    ws.views = [{ state: "frozen", ySplit: 1 }];
}

async function downloadExcelOld() {
    if (typeof ExcelJS === "undefined") {
        batchStatus.textContent = "The Excel library could not load. Check your internet connection and try again.";
        return;
    }
    if (!batchRows.length) return;

    const hasRf = batchRows[0].pRf !== null;
    const inputCols = borrowerColumns();
    const cols = ["No"].concat(inputCols,
        ["pd_lr", "skor_scorecard_lr", "skor_lr", "risk_level_lr"],
        hasRf ? ["pd_rf", "skor_rf", "risk_level_rf"] : []);

    const wb = new ExcelJS.Workbook();
    const ws = wb.addWorksheet("Hasil");
    ws.addRow(cols);
    batchRows.forEach(r => {
        const row = [r.no].concat(
            inputCols.map(c => (r.vals[c] !== undefined ? r.vals[c] : r.cats[c])),
            [r.p, r.scorecard, r.s, r.level.key]);
        if (hasRf) row.push(r.pRf, r.sRf, r.levelRf.key);
        ws.addRow(row);
    });
    
    styleHeader(ws);
    ws.columns.forEach(c => { c.width = 18; });

    const col = name => cols.indexOf(name) + 1;
    ["pd_lr", "pd_rf"].forEach(n => { if (col(n) > 0) ws.getColumn(col(n)).numFmt = "0.0000"; });
    ["skor_scorecard_lr", "skor_lr", "skor_rf"].forEach(n => { if (col(n) > 0) ws.getColumn(col(n)).numFmt = "0.0"; });
    ["risk_level_lr", "risk_level_rf"].forEach(n => {
        const ci = col(n);
        if (ci < 1) return;
        for (let i = 2; i <= batchRows.length + 1; i++) {
            const cell = ws.getRow(i).getCell(ci);
            cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: LEVEL_FILL[cell.value] } };
        }
    });

    if (hasilData) {
        const wm = wb.addWorksheet("Perbandingan_Model");
        wm.addRow(["model", "akurasi", "presisi", "recall", "f1", "AUC", "Gini", "KS"]);
        ["lr", "rf"].forEach(k => {
            const m = hasilData.metrik[k];
            wm.addRow([m.model, m.akurasi, m.presisi, m.recall, m.f1, m.AUC, m.Gini, m.KS]);
        });
        styleHeader(wm);
        wm.columns.forEach(c => { c.width = 20; });

        const wk = wb.addWorksheet("Koefisien_LR");
        wk.addRow(["fitur", "koefisien", "rasio_odds"]);
        hasilData.koefisien.forEach(x => wk.addRow([x.fitur, x.koefisien, x.rasio_odds]));
        styleHeader(wk);
        wk.columns.forEach(c => { c.width = 34; });

        const wi = wb.addWorksheet("Importance_RF");
        wi.addRow(["fitur", "importance"]);
        hasilData.importance.forEach(x => wi.addRow([x.fitur, x.importance]));
        styleHeader(wi);
        wi.columns.forEach(c => { c.width = 34; });
    }

    const buf = await wb.xlsx.writeBuffer();
    downloadFile("credit_scoring_results.xlsx", buf,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
}

excelBtn.addEventListener("click", downloadExcel);



/* ---------- RESULTS TABLE WITH BORROWER DATA ---------- */
// same colour bands as the PD cells in the Excel sheet
function pdBand(p) { return p < 0.10 ? "low" : p < 0.30 ? "mid" : "high"; }

function pdCell(p) {
    return '<td><span class="pd-chip pd-' + pdBand(p) + '">' + formatPersen(p) + "</span></td>";
}

function levelCell(level) {
    return '<td><span class="lvl-chip ' + level.cls + '">' + level.label + "</span></td>";
}

function resultsRow(r) {
    const v = r.vals;
    return "<tr><td>" + escHtml(r.no) + "</td><td>" + v.person_age + "</td><td>" + v.person_emp_length +
        "</td><td>" + fmtInt(v.person_income) + "</td><td>" + r.cats.person_home_ownership +
        "</td><td>" + fmtInt(v.loan_amnt) + "</td><td>" + Number(v.loan_int_rate.toFixed(2)) +
        "</td><td>" + v.cb_person_cred_hist_length + "</td><td>" + (v.cb_person_default_on_file_code ? "Yes" : "No") + "</td>" +
        pdCell(r.p) + "<td>" + Math.round(r.s) + "</td>" + levelCell(r.level) +
        (r.pRf === null ? "<td>–</td><td>–</td><td>–</td>"
            : pdCell(r.pRf) + "<td>" + Math.round(r.sRf) + "</td>" + levelCell(r.levelRf)) +
        "</tr>";
}

function resultsCard(rows) {
    const top = rows.slice().sort((a, b) => b.p - a.p).slice(0, 10);
    const heads = ["Borrower", "Age (years)", "Employment Length (years)", "Annual Income", "Home Ownership",
        "Loan Amount", "Interest Rate (%)", "Credit History Length (years)", "Previous Default on File",
        "Probability of Default (Logistic Regression)", "Credit Score (Logistic Regression)", "Risk Level (Logistic Regression)",
        "Probability of Default (Random Forest)", "Credit Score (Random Forest)", "Risk Level (Random Forest)"];
    return '<div class="mc-chart-card"><h3>Highest-risk borrowers</h3>' +
        '<p class="mc-caption">The ten borrowers with the highest probability of default under the Logistic Regression, ' +
        "shown with their data so you can tell who they are. Download the Excel file for every borrower and every column of your file. " +
        "Colours: green below 10%, yellow from 10% to 30%, red from 30%.</p>" +
        '<div class="mc-table-wrap"><table class="mc-result-table mc-wide-table"><thead><tr>' +
        heads.map(h => "<th>" + h + "</th>").join("") + "</tr></thead><tbody>" +
        top.map(resultsRow).join("") + "</tbody></table></div></div>";
}

/* ---------- EXCEL DOWNLOAD (full version with borrower data) ---------- */
const PD_FILL = { low: "FFC6EFCE", mid: "FFFFEB9C", high: "FFFFC7CE" };
const PD_FONT = { low: "FF006100", mid: "FF9C5700", high: "FF9C0006" };

function colourHeader(ws, from, to, argb) {
    for (let c = from; c <= to; c++) {
        const cell = ws.getRow(1).getCell(c);
        cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb } };
        cell.font = { bold: true, color: { argb: "FF1F1F1F" } };
        cell.alignment = { wrapText: true, vertical: "middle" };
    }
}

async function downloadExcel() {
    if (typeof ExcelJS === "undefined") {
        batchStatus.textContent = "The Excel library could not load. Check your internet connection and try again.";
        return;
    }
    if (!batchRows.length) return;

    const hasRf = batchRows[0].pRf !== null;
    const head = batchFileData.head.map((h, i) => h || "Column " + (i + 1));
    const lrCols = ["Probability of Default (Logistic Regression)", "Credit Score 300-850 (Logistic Regression)",
        "Scorecard Score (Logistic Regression)", "Risk Level (Logistic Regression)", "Predicted Status (Logistic Regression)"];
    const rfCols = hasRf ? ["Probability of Default (Random Forest)", "Credit Score 300-850 (Random Forest)",
        "Risk Level (Random Forest)", "Predicted Status (Random Forest)"] : [];

    const wb = new ExcelJS.Workbook();
    const ws = wb.addWorksheet("Credit Scoring Results");
    ws.addRow(["Borrower Number"].concat(head, lrCols, rfCols));

    const toCell = x => {
        const t = String(x == null ? "" : x).trim();
        return /^-?\d+(\.\d+)?$/.test(t) ? Number(t) : t;
    };
    const status = p => (p >= ambang ? "Default" : "No default");

    batchRows.forEach((r, i) => {
        const row = [i + 1].concat(head.map((_, k) => toCell(r.raw[k])),
            [r.p, r.s, r.scorecard, r.level.label, status(r.p)]);
        if (hasRf) row.push(r.pRf, r.sRf, r.levelRf.label, status(r.pRf));
        ws.addRow(row);
    });

    const lr = 2 + head.length, rfc = lr + 5;   // first column of each result group
    batchRows.forEach((r, i) => {
        const row = ws.getRow(i + 2);
        const paint = (col, fill, font) => {
            const c = row.getCell(col);
            c.fill = { type: "pattern", pattern: "solid", fgColor: { argb: fill } };
            if (font) c.font = { color: { argb: font } };
        };
        const b1 = pdBand(r.p);
        paint(lr, PD_FILL[b1], PD_FONT[b1]);
        paint(lr + 3, LEVEL_FILL[r.level.key]);
        if (hasRf) {
            const b2 = pdBand(r.pRf);
            paint(rfc, PD_FILL[b2], PD_FONT[b2]);
            paint(rfc + 2, LEVEL_FILL[r.levelRf.key]);
        }
    });

    ws.getColumn(lr).numFmt = "0.0%";
    ws.getColumn(lr + 1).numFmt = "0";
    ws.getColumn(lr + 2).numFmt = "0";
    if (hasRf) { ws.getColumn(rfc).numFmt = "0.0%"; ws.getColumn(rfc + 1).numFmt = "0"; }

    colourHeader(ws, 1, 1 + head.length, "FFBDD7EE");          // borrower data: blue
    colourHeader(ws, lr, lr + 4, "FFC6E0B4");                  // Logistic Regression: green
    if (hasRf) colourHeader(ws, rfc, rfc + 3, "FFF8CBAD");     // Random Forest: orange
    ws.getRow(1).height = 48;
    ws.columns.forEach((c, k) => { c.width = k >= lr - 1 ? 24 : 16; });
    ws.views = [{ state: "frozen", xSplit: 1, ySplit: 1 }];

    if (hasilData) {
        const wm = wb.addWorksheet("Model Comparison");
        wm.addRow(["Model", "Accuracy", "Precision", "Recall", "F1 Score",
            "AUC (Area Under ROC Curve)", "Gini Coefficient", "KS Statistic"]);
        ["lr", "rf"].forEach(k => {
            const m = hasilData.metrik[k];
            wm.addRow([m.model, m.akurasi, m.presisi, m.recall, m.f1, m.AUC, m.Gini, m.KS]);
        });
        styleHeader(wm);
        wm.columns.forEach(c => { c.width = 24; });

        const wk = wb.addWorksheet("Logistic Coefficients");
        wk.addRow(["Feature", "Coefficient", "Odds Ratio"]);
        hasilData.koefisien.forEach(x => wk.addRow([featLabel(x.fitur), x.koefisien, x.rasio_odds]));
        styleHeader(wk);
        wk.columns.forEach(c => { c.width = 30; });

        const wi = wb.addWorksheet("Random Forest Importance");
        wi.addRow(["Feature", "Importance"]);
        hasilData.importance.forEach(x => wi.addRow([featLabel(x.fitur), x.importance]));
        styleHeader(wi);
        wi.columns.forEach(c => { c.width = 30; });
    }

    const buf = await wb.xlsx.writeBuffer();
    downloadFile("credit_scoring_results.xlsx", buf,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
}