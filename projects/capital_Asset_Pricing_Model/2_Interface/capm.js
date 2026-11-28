
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
const KODE_URL = "../1_Model/CAPM.py";
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
            "Could not load CAPM.py. Open this page with Live Server " +
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
        document.getElementById("mProfit").textContent = formatPersen(h.peluang_untung);
        document.getElementById("mLoss").textContent = formatPersen(h.peluang_rugi);
        document.getElementById("mVar").textContent = formatRupiah(h.var_95);
        document.getElementById("mVarNote").textContent =
        formatPersen(h.var_95 / h.modal_awal) + " of the initial capital";
        const setText = (id, text) => { document.getElementById(id).textContent = text; };
        setText("mEs", formatRupiah(h.es_95));
        setText("mEsNote", formatPersen(h.es_95 / h.modal_awal) + " of the initial capital");
        setText("mAvg", formatRupiah(h.rata_akhir));
        setText("mMed", formatRupiah(h.median_akhir));
        setText("mP5", formatRupiah(h.persentil_5));
        setText("mP95", formatRupiah(h.persentil_95));
        setText("mKey", "Key finding: a " + formatPersen(h.peluang_untung) +
            " chance of profit, but the worst 5% of scenarios lose " +
            formatRupiah(h.es_95) + " on average.");
        document.getElementById("mMeta").textContent =
            h.saham.join(", ") + " · " + h.periode[0] + " to " + h.periode[1] + " · " +
            h.jumlah_simulasi.toLocaleString("en-US") + " simulations · " +
            h.jumlah_hari + " trading days";
    })
    .catch(() => {
        document.getElementById("mMeta").textContent =
            "Results file not found yet. Run the Python script once to create it.";
    });

 /* ---------- TAB 3: UPLOAD CSV & DATA ---------- */
const stocks = new Map(); // nama -> {dates, closes, priceCol}
const MAX_FILES = 10;
const MONTHS = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, mei: 5, jun: 6, jul: 7,
                 aug: 8, agu: 8, sep: 9, oct: 10, okt: 10, nov: 11, dec: 12, des: 12 };

const csvInput = document.getElementById("csvFiles");
const dropZone = document.getElementById("dropZone");
const fileListEl = document.getElementById("fileList");
const csvStatus = document.getElementById("csvStatus");
const runButton = document.getElementById("runSimulation");
const simResults = document.getElementById("simResults");

const esc = s => s.replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const iso = (y, m, d) => y + "-" + String(m).padStart(2, "0") + "-" + String(d).padStart(2, "0");

// memecah satu baris CSV dengan memperhatikan tanda kutip
function splitLine(line, sep) {
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

// "5,250.00" / "5.250,00" / "5250" -> angka
function parseNumber(raw) {
    let s = raw.replace(/[^\d.,-]/g, "");
    if (!s) return NaN;
    const hasDot = s.includes("."), hasComma = s.includes(",");
    if (hasDot && hasComma) {
        s = s.lastIndexOf(",") > s.lastIndexOf(".")
            ? s.replace(/\./g, "").replace(",", ".")
            : s.replace(/,/g, "");
    } else if (hasComma) {
        s = /^-?\d{1,3}(,\d{3})+$/.test(s) ? s.replace(/,/g, "") : s.replace(",", ".");
    } else if (hasDot && /^-?\d{1,3}(\.\d{3})+$/.test(s)) {
        s = s.replace(/\./g, "");
    }
    return parseFloat(s);
}

// semua format tanggal -> YYYY-MM-DD
function parseDates(rawList) {
    let dayFirst = rawList.some(r => /^\d{1,2}\.\d{1,2}\.\d{4}$/.test(r));
    for (const r of rawList) {
        const m = r.match(/^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{4})$/);
        if (!m) continue;
        if (+m[1] > 12) { dayFirst = true; break; }
        if (+m[2] > 12) { dayFirst = false; break; }
    }
    return rawList.map(r => {
        let m = r.match(/^(\d{4})[\/.-](\d{1,2})[\/.-](\d{1,2})/);
        if (m) return iso(m[1], m[2], m[3]);
        m = r.match(/^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{4})$/);
        if (m) return dayFirst ? iso(m[3], m[2], m[1]) : iso(m[3], m[1], m[2]);
        m = r.match(/^([A-Za-z]{3})[a-z]*\.? (\d{1,2}),? (\d{4})$/);
        if (m && MONTHS[m[1].toLowerCase()]) return iso(m[3], MONTHS[m[1].toLowerCase()], m[2]);
        m = r.match(/^(\d{1,2}) ([A-Za-z]{3})[a-z]*\.? (\d{4})$/);
        if (m && MONTHS[m[2].toLowerCase()]) return iso(m[3], MONTHS[m[2].toLowerCase()], m[1]);
        return null;
    });
}

function parseCsv(text) {
    const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(l => l.trim());
    if (lines.length < 3) throw new Error("file is empty");

    const commas = (lines[0].match(/,/g) || []).length;
    const semis = (lines[0].match(/;/g) || []).length;
    const sep = semis > commas ? ";" : ",";
    const head = splitLine(lines[0], sep).map(h => h.toLowerCase());

    const di = head.findIndex(h => h === "date" || h === "tanggal");
    let pi = -1;
    for (const name of ["adj close", "adj_close", "close", "price", "terakhir", "penutupan", "last"]) {
        pi = head.indexOf(name);
        if (pi >= 0) break;
    }
    if (di < 0 || pi < 0) {
        throw new Error("Date or Close column not found (columns: " + head.join(", ") + ")");
    }

    const rows = lines.slice(1).map(l => splitLine(l, sep));
    const dates = parseDates(rows.map(r => r[di] || ""));
    const map = new Map();
    rows.forEach((r, i) => {
        const p = parseNumber(r[pi] || "");
        if (dates[i] && p > 0) map.set(dates[i], p);
    });
    if (map.size < 30) {
        throw new Error("only " + map.size + " valid rows read. First data row: " + lines[1].slice(0, 70));
    }

    const sorted = Array.from(map.keys()).sort();
    return { dates: sorted, closes: sorted.map(d => map.get(d)), priceCol: head[pi] };
}

function cleanName(file) {
    const n = file.name.replace(/\.csv$/i, "")
        .replace(/stock price history|historical data|riwayat harga saham|data historis|\(\d+\)/gi, "")
        .replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
    return (n || file.name).slice(0, 28);
}

// tanggal yang dimiliki semua saham
function commonDates() {
    const names = Array.from(stocks.keys());
    if (!names.length) return [];
    const sets = names.map(n => new Set(stocks.get(n).dates));
    return stocks.get(names[0]).dates.filter(d => sets.every(s => s.has(d)));
}

// log-return harian pada periode terpilih (dipakai juga di tahap 2)
function buildReturns(years) {
    const names = Array.from(stocks.keys());
    const price = names.map(n => {
        const s = stocks.get(n);
        return new Map(s.dates.map((d, i) => [d, s.closes[i]]));
    });
    let dates = commonDates();
    if (!dates.length) return { names, dates, returns: [] };
    const last = dates[dates.length - 1];
    const cutoff = (parseInt(last.slice(0, 4), 10) - years) + last.slice(4);
    dates = dates.filter(d => d >= cutoff);
    const returns = price.map(p =>
        dates.slice(1).map((d, i) => Math.log(p.get(d) / p.get(dates[i]))));
    return { names, dates, returns };
}

function renderFiles(errors) {
    fileListEl.innerHTML = "";
    stocks.forEach((s, name) => {
        const li = document.createElement("li");
        li.innerHTML = '<div><strong></strong><small></small></div>' +
                       '<button type="button" aria-label="Remove">×</button>';
        li.querySelector("strong").textContent = name;
        li.querySelector("small").textContent =
            s.dates.length + " days · " + s.dates[0] + " to " +
            s.dates[s.dates.length - 1] + " · column \"" + s.priceCol + "\"";
        li.querySelector("button").addEventListener("click", () => {
            stocks.delete(name);
            renderFiles([]);
        });
        fileListEl.appendChild(li);
    });

    const common = commonDates().length;
    const ready = stocks.size > 0 && common >= 61;
    runButton.disabled = !ready;

    let msg;
    if (!stocks.size) {
        msg = "No files yet. The file needs a Date column and a Close / Price column.";
    } else if (ready) {
        msg = stocks.size + " stock(s) ready · " + common + " common trading days.";
    } else {
        msg = "Only " + common + " common trading days (need at least 61). " +
              "Make sure all files cover the same period.";
    }
    if (errors.length) msg += " Problem: " + errors.join(" | ");
    csvStatus.textContent = msg;
}

async function addFiles(files) {
    const errors = [];
    for (const file of Array.from(files)) {
        if (stocks.size >= MAX_FILES) { errors.push("maximum " + MAX_FILES + " files"); break; }
        if (file.size > 5 * 1024 * 1024) { errors.push(file.name + ": larger than 5 MB"); continue; }
        try {
            const data = parseCsv(await file.text());
            const base = cleanName(file);
            let name = base, k = 2;
            while (stocks.has(name)) name = base + " " + k++;
            stocks.set(name, data);
        } catch (err) {
            errors.push(file.name + ": " + err.message);
        }
    }
    renderFiles(errors);
}

csvInput.addEventListener("change", event => {
    addFiles(event.target.files);
    csvInput.value = "";
});

["dragenter", "dragover"].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev =>
    dropZone.addEventListener(ev, e => { e.preventDefault(); dropZone.classList.remove("dragover"); }));
dropZone.addEventListener("drop", e => addFiles(e.dataTransfer.files));

// sementara: tombol Run menampilkan ringkasan data (mesin simulasi = tahap 2)
/* ---------- TAB 3: MESIN SIMULASI ---------- */
let confLevel = 0.95;
document.querySelectorAll(".mc-seg button").forEach(btn => {
    btn.addEventListener("click", () => {
        confLevel = parseFloat(btn.dataset.conf);
        document.querySelectorAll(".mc-seg button")
            .forEach(b => b.classList.toggle("on", b === btn));
    });
});

const pct = x => (x < 0 ? "−" : "") + Math.abs(x * 100).toFixed(1) + "%";
const pct0 = x => (x * 100).toFixed(0) + "%";
const rpShort = x => x >= 1e9
    ? "Rp " + (x / 1e9).toFixed(1) + "B"
    : "Rp " + (x / 1e6).toFixed(1) + "M";

function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

// persentil dengan interpolasi linear (sama seperti np.percentile)
function pctile(sorted, p) {
    const i = p * (sorted.length - 1), lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function runMonteCarlo(daily, nSims, horizon, conf) {
    const n = daily.length;
    const mu = daily.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(daily.reduce((a, b) => a + (b - mu) ** 2, 0) / (n - 1));

    const finals = new Float64Array(nSims);
    const meanPath = new Float64Array(horizon + 1);
    const sample = [];
    const grid = new Float32Array(nSims * (horizon + 1));

    for (let s = 0; s < nSims; s++) {
        let v = 1;
        const keep = s < 1000 ? [1] : null;
        meanPath[0] += 1;
        grid[s * (horizon + 1)] = 1;
        for (let t = 1; t <= horizon; t++) {
            v *= 1 + mu + sd * randn();
            if (v < 0) v = 0;
            meanPath[t] += v;
            grid[s * (horizon + 1) + t] = v;
            if (keep) keep.push(v);
        }
        finals[s] = v;
        if (keep) sample.push(keep);
    }
    for (let t = 0; t <= horizon; t++) meanPath[t] /= nSims;
    const p5Path = [], p95Path = [];
    for (let t = 0; t <= horizon; t++) {
    const col = new Float64Array(nSims);
        for (let s = 0; s < nSims; s++) col[s] = grid[s * (horizon + 1) + t];
        col.sort();
        p5Path.push(pctile(col, 0.05));
        p95Path.push(pctile(col, 0.95));
}

    const rets = Array.from(finals, v => v - 1).sort((a, b) => a - b);
    const q = 1 - conf;
    const varR = pctile(rets, q);
    const tail = rets.filter(x => x <= varR);
    const es = tail.reduce((a, b) => a + b, 0) / tail.length;
    const pProfit = rets.filter(x => x > 0).length / nSims;

    const histSorted = daily.slice().sort((a, b) => a - b);
        return { rets, varR, es, pProfit, sample, meanPath, p5Path, p95Path, histVar: pctile(histSorted, q) };
}


function histSvg(values, split, label) {
    const bins = 40, W = 400, H = 170, pad = 18;
    let min = Infinity, max = -Infinity;
    values.forEach(v => { if (v < min) min = v; if (v > max) max = v; });
    const w = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    values.forEach(v => counts[Math.min(bins - 1, Math.floor((v - min) / w))]++);
    const top = Math.max(...counts), bw = (W - 2 * pad) / bins;

    let bars = "";
    counts.forEach((c, i) => {
        const h = c / top * (H - 50);
        const center = min + (i + 0.5) * w;
        bars += '<rect x="' + (pad + i * bw + 1).toFixed(1) + '" y="' + (H - 20 - h).toFixed(1) +
                '" width="' + (bw - 2).toFixed(1) + '" height="' + h.toFixed(1) +
                '" fill="' + (center < split ? "#d4a5a0" : "#682127") + '"/>';
    });
    const frac = Math.max(0, Math.min(1, (split - min) / (max - min || 1)));
    const x = pad + frac * (W - 2 * pad);
    const anchor = frac > 0.7 ? "end" : "start";
    const tx = anchor === "end" ? x - 6 : x + 6;
    return '<svg viewBox="0 0 ' + W + " " + H + '">' + bars +
        '<line x1="' + pad + '" x2="' + (W - pad) + '" y1="' + (H - 20) + '" y2="' + (H - 20) + '" stroke="#806d68"/>' +
        '<line x1="' + x.toFixed(1) + '" x2="' + x.toFixed(1) + '" y1="16" y2="' + (H - 20) +
        '" stroke="#682127" stroke-width="2" stroke-dasharray="4 3"/>' +
        '<text x="' + tx.toFixed(1) + '" y="13" font-size="11" fill="#682127" text-anchor="' + anchor + '">' +
        label + "</text></svg>";
}


const PATH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

function pathsSvg(sample, meanPath, p5, p95, capital) {
    const W = 800, H = 290, L = 56, P = 12, B = 28, T = meanPath.length - 1;
    let lo = 1, hi = 1;
    sample.forEach(p => p.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    meanPath.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; });
    [p5, p95].forEach(a => a.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));

    const X = t => L + t / T * (W - L - P);
    const Y = v => H - B - (v - lo) / ((hi - lo) || 1) * (H - B - P);
    let axis = "";
    for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4;
    axis += '<text x="' + (L - 8) + '" y="' + (Y(v) + 4).toFixed(1) +
    '" font-size="11" fill="#806d68" text-anchor="end">' +
    (v * capital / 1e6).toFixed(1) + "</text>";
    }

    const line = (arr, sw, col) =>
        '<polyline fill="none" stroke="' + col + '" stroke-width="' + sw + '" points="' +
        Array.from(arr, (v, t) => X(t).toFixed(1) + "," + Y(v).toFixed(1)).join(" ") + '"/>';
    return '<svg viewBox="0 0 ' + W + " " + H + '">' +
            sample.map((p, k) =>
            '<polyline fill="none" stroke="' + PATH_COLORS[k % 10] +
            '" stroke-opacity="0.35" stroke-width="0.8" points="' +
            p.map((v, t) => X(t).toFixed(1) + "," + Y(v).toFixed(1)).join(" ") + '"/>').join("") +
    
            '<line x1="' + L + '" x2="' + (W - P) + '" y1="' + Y(1).toFixed(1) + '" y2="' + Y(1).toFixed(1) +
            '" stroke="#806d68" stroke-dasharray="5 4"/>' +
            line(p95, 2.5, "#2e8b3d").replace("<polyline", '<polyline stroke-dasharray="7 5"') +
            line(p5, 2.5, "#d62728").replace("<polyline", '<polyline stroke-dasharray="7 5"') +
            line(meanPath, 3, "#682127") + axis +
            '<text x="' + L + '" y="' + (H - 8) + '" font-size="11" fill="#806d68">Day 0</text>' +
            '<text x="' + (W - P) + '" y="' + (H - 8) + '" font-size="11" fill="#806d68" text-anchor="end">Day ' + T + "</text>" +
            '<text x="' + (L + 6) + '" y="11" font-size="11" fill="#806d68">Portfolio value (Rp millions)</text></svg>';
            }


function histFinalSvg(rets, capital, money) {
    const bins = 60, W = 700, H = 300, L = 14, P = 14, B = 34, base = H - B;
    const v = rets.map(x => (1 + x) * capital);   // sudah terurut naik
    const min = v[0], max = v[v.length - 1];
    const w = (max - min) / bins || 1;
    const counts = new Array(bins).fill(0);
    v.forEach(x => counts[Math.min(bins - 1, Math.floor((x - min) / w))]++);
    const top = Math.max(...counts), bw = (W - L - P) / bins;
    const X = x => L + (x - min) / ((max - min) || 1) * (W - L - P);

    let bars = "";
    counts.forEach((c, i) => {
        const h = c / top * (base - 24);
        const center = min + (i + 0.5) * w;
        bars += '<rect x="' + (L + i * bw + 0.5).toFixed(1) + '" y="' + (base - h).toFixed(1) +
                '" width="' + (bw - 1).toFixed(1) + '" height="' + h.toFixed(1) +
                '" fill="' + (center < capital ? "#d4a5a0" : "#682127") + '"/>';
    });

    const vline = (x, col, sw, dash) =>
        '<line x1="' + X(x).toFixed(1) + '" x2="' + X(x).toFixed(1) + '" y1="18" y2="' + base +
        '" stroke="' + col + '" stroke-width="' + sw + '"' +
        (dash ? ' stroke-dasharray="' + dash + '"' : "") + "/>";

    let ticks = "";
    for (let k = 0; k <= 5; k++) {
        const x = min + (max - min) * k / 5;
        ticks += '<text x="' + X(x).toFixed(1) + '" y="' + (H - 12) +
                 '" font-size="11" fill="#806d68" text-anchor="' +
                 (k === 0 ? "start" : k === 5 ? "end" : "middle") + '">' + (x / 1e6).toFixed(1) + "</text>";
    }

    return '<svg viewBox="0 0 ' + W + " " + H + '">' + bars +
        '<line x1="' + L + '" x2="' + (W - P) + '" y1="' + base + '" y2="' + base + '" stroke="#806d68"/>' +
        vline(capital, "#806d68", 2, "2 4") +
        vline(money.median, "#222222", 2.5) +
        vline(money.mean, "#e08a1e", 2.5) +
        vline(money.p5, "#d62728", 2, "6 4") +
        vline(money.p95, "#2e8b3d", 2, "6 4") +
        ticks +
        '<text x="' + (L + 4) + '" y="11" font-size="11" fill="#806d68">Final portfolio value (Rp millions)</text></svg>';
}
   
function barPanel(title, vals, names, fmt) {
    const W = 300, H = 234, L = 12, R = 12, T = 34, B = 40, n = vals.length;
    const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    const span = (hi - lo) || 1;
    const Y = v => T + (hi - v) / span * (H - T - B);
    const slot = (W - L - R) / n, bw = Math.min(46, slot * 0.7);

    let out = '<svg viewBox="0 0 ' + W + " " + H + '">' +
        '<text x="' + (W / 2) + '" y="16" font-size="12" fill="#3a1e20" text-anchor="middle">' + title + "</text>" +
        '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(0).toFixed(1) + '" y2="' + Y(0).toFixed(1) + '" stroke="#806d68"/>';

    vals.forEach((v, i) => {
        const cx = L + slot * (i + 0.5);
        const top = Y(Math.max(v, 0)), h = Math.abs(Y(v) - Y(0));
        out += '<rect x="' + (cx - bw / 2).toFixed(1) + '" y="' + top.toFixed(1) + '" width="' + bw.toFixed(1) +
               '" height="' + Math.max(h, 1).toFixed(1) + '" fill="' + PATH_COLORS[i % 10] + '"/>' +
               '<text x="' + cx.toFixed(1) + '" y="' + (v >= 0 ? top - 5 : Y(v) + 14).toFixed(1) +
               '" font-size="11" fill="#3a1e20" text-anchor="middle">' + fmt(v) + "</text>" +
               '<text x="' + cx.toFixed(1) + '" y="' + (H - 8) + '" font-size="10" fill="#806d68" text-anchor="middle">' +
               esc(names[i].slice(0, 7)) + "</text>";
    });
    return out + "</svg>";
}

function riskScaleSvg(varL, esL, worstL) {
    if (worstL <= 0) return "";
    const W = 700, H = 112, L = 24, R = 24, y = 54;
    const X = l => L + Math.min(l, worstL) / worstL * (W - L - R);
    const anchor = l => { const f = l / worstL; return f > 0.8 ? "end" : f < 0.2 ? "start" : "middle"; };
    const dx = l => { const f = l / worstL; return f > 0.8 ? -4 : f < 0.2 ? 4 : 0; };

    return '<svg viewBox="0 0 ' + W + " " + H + '">' +
        '<rect x="' + X(0).toFixed(1) + '" y="' + (y - 8) + '" width="' + (X(varL) - X(0)).toFixed(1) + '" height="16" fill="#e8d3cd"/>' +
        '<rect x="' + X(varL).toFixed(1) + '" y="' + (y - 8) + '" width="' + (X(worstL) - X(varL)).toFixed(1) + '" height="16" fill="#c98a85"/>' +
        '<line x1="' + X(varL).toFixed(1) + '" x2="' + X(varL).toFixed(1) + '" y1="' + (y - 22) + '" y2="' + (y + 12) +
        '" stroke="#682127" stroke-width="2" stroke-dasharray="4 3"/>' +
        '<text x="' + (X(varL) + dx(varL)).toFixed(1) + '" y="' + (y - 28) + '" font-size="12" fill="#682127" text-anchor="' + anchor(varL) +
        '">VaR ' + pct(-varL) + "</text>" +
        '<line x1="' + X(esL).toFixed(1) + '" x2="' + X(esL).toFixed(1) + '" y1="' + (y - 12) + '" y2="' + (y + 22) +
        '" stroke="#b4564f" stroke-width="2.5"/>' +
        '<text x="' + (X(esL) + dx(esL)).toFixed(1) + '" y="' + (y + 38) + '" font-size="12" fill="#b4564f" text-anchor="' + anchor(esL) +
        '">ES ' + pct(-esL) + "</text>" +
        '<text x="' + L + '" y="' + (H - 4) + '" font-size="11" fill="#806d68">No loss</text>' +
        '<text x="' + (W - R) + '" y="' + (H - 4) + '" font-size="11" fill="#806d68" text-anchor="end">Worst path ' + pct(-worstL) + "</text></svg>";
}

function renderResults(r, d, o) {
    const cl = Math.round(o.conf * 100), tailPct = 100 - cl;
    const lossP = 1 - r.pProfit;
    const nProfit = Math.round(r.pProfit * o.nSims), nLoss = o.nSims - nProfit;
    const fmtN = x => x.toLocaleString("en-US");
    const varLoss = Math.max(0, -r.varR);
    const lastDay = r.meanPath.length - 1;
    const money = {
        mean: r.meanPath[lastDay] * o.capital,
        median: (1 + pctile(r.rets, 0.5)) * o.capital,
        p5: r.p5Path[lastDay] * o.capital,
        p95: r.p95Path[lastDay] * o.capital
    };

    const esLoss = Math.max(0, -r.es);
    const worstLoss = Math.max(0, -r.rets[0]);

        const rows = d.stats.map(s =>
        "<tr><td>" + esc(s.name) + "</td><td>" + pct(s.ret) + "</td><td>" + pct(s.vol) + "</td><td>" +
        s.sharpe.toFixed(2) + "</td><td>" + pct(s.w) + "</td></tr>").join("");

    let advProfit;
    if (r.pProfit >= 0.6) {
        advProfit = "<strong>" + pct0(r.pProfit) + " chance of profit.</strong> Attractive but not safe. " +
            "Consider entering in stages and set a stop-loss level before you buy.";
    } else if (r.pProfit >= 0.5) {
        advProfit = "<strong>" + pct0(r.pProfit) + " chance of profit.</strong> Only slightly better than a " +
            "coin flip. Look closely at the downside before committing money.";
    } else {
        advProfit = "<strong>Only " + pct0(r.pProfit) + " chance of profit.</strong> Fewer than half of the " +
            "paths end in a gain. Consider a smaller position or a different stock.";
    }

    const diversify = d.names.length === 1
        ? "<strong>Diversify.</strong> Don't put all your capital into one stock, since this simulation only evaluates a single stock."
        : "<strong>Diversify wisely.</strong> This portfolio uses Markowitz weights across" + d.names.length +
          " stocks. Diversification lowers risk but does not remove it, especially when the stocks move together.";

    const varText = r.varR < 0
        ? "In " + cl + "% of scenarios, your loss is no worse than " + pct(varLoss).replace("−", "") +
          ". In the remaining " + tailPct + "% it can be worse."
        : "Even the worst " + tailPct + "% of scenarios stay at or above the initial capital (VaR is " +
          pct(r.varR) + "). Real markets can still behave differently from this model.";

    const html =
        '<div class="mc-res-head"><h3>Results for ' + esc(d.names.join(", ")) + "</h3><span>" +
        fmtN(o.nSims) + " simulations · " + o.horizon + " trading days · " + cl + "% confidence</span></div>" +

        '<div class="mc-metrics">' +
        '<article class="mc-metric"><span>Probability of profit</span><strong>' + pct0(r.pProfit) +
        "</strong><small>paths ending above capital</small></article>" +
        '<article class="mc-metric"><span>Probability of loss</span><strong class="neg">' + pct0(lossP) +
        "</strong><small>paths ending below capital</small></article>" +
        '<article class="mc-metric"><span>VaR ' + cl + '%</span><strong>' + pct(r.varR) +
        "</strong><small>≈ " + formatRupiah(varLoss * o.capital) + " of " + formatRupiah(o.capital) + "</small></article>" +
        '<article class="mc-metric"><span>Expected Shortfall</span><strong>' + pct(r.es) +
        "</strong><small>average of the worst " + tailPct + "%</small></article></div>" +
                 '<div class="mc-metrics">' +
        '<article class="mc-metric"><span>Average final value</span><strong class="money">' + formatRupiah(money.mean) +
        "</strong><small>mean of all paths</small></article>" +
        '<article class="mc-metric"><span>Median final value</span><strong class="money">' + formatRupiah(money.median) +
        "</strong><small>half of the paths end above</small></article>" +
        '<article class="mc-metric"><span>Bad scenario (P5)</span><strong class="money neg">' + formatRupiah(money.p5) +
        "</strong><small>only 5% of paths end lower</small></article>" +
        '<article class="mc-metric"><span>Good scenario (P95)</span><strong class="money">' + formatRupiah(money.p95) +
        "</strong><small>only 5% of paths end higher</small></article></div>" +
        
        '<div class="mc-chart-card"><h3>Stocks and Markowitz weights</h3>' +
        '<table class="mc-result-table"><thead><tr><th>STOCK</th><th>RETURN / YEAR</th>' +
        "<th>VOLATILITY / YEAR</th><th>SHARPE</th><th>WEIGHT</th></tr></thead><tbody>" + rows + "</tbody></table>" +
        "<p>Weights maximize the Sharpe ratio (risk-free rate 6%, max " + pct0(d.cap) + " per stock).</p></div>" +
        '<div class="mc-chart-card"><h3>Stock comparison</h3>' +
        "<p>How each stock contributes. Colors follow the stock order in the table above.</p>" +
        '<div class="mc-bars">' +
        barPanel("Annual return (%)", d.stats.map(s => s.ret * 100), d.names, v => v.toFixed(1)) +
        barPanel("Annual volatility (%)", d.stats.map(s => s.vol * 100), d.names, v => v.toFixed(1)) +
        barPanel("Sharpe ratio", d.stats.map(s => s.sharpe), d.names, v => v.toFixed(2)) +
        barPanel("Markowitz weight (%)", d.stats.map(s => s.w * 100), d.names, v => v.toFixed(1)) +
        "</div></div>" +
        '<div class="mc-chart-card"><h3>Simulated price paths</h3><p>A sample of the ' + fmtN(Math.min(o.nSims, 1000)) +
        " paths. The thick line is the average; the dashed line is the initial capital.</p>" +
        pathsSvg(r.sample, r.meanPath, r.p5Path, r.p95Path, o.capital) +
        '<div class="mc-legend">' +
        '<span><i style="border-color:#682127"></i>Average</span>' +
        '<span><i class="dash" style="border-color:#d62728"></i>5th percentile</span>' +
        '<span><i class="dash" style="border-color:#2e8b3d"></i>95th percentile</span>' +
        '<span><i class="dash" style="border-color:#806d68"></i>Initial capital</span></div></div>' +

        '<div class="mc-res-two">' +
        '<div class="mc-chart-card"><h3>Historical VaR</h3><p>Distribution of daily returns from historical data.</p>' +
        histSvg(d.daily, r.histVar, "VaR " + cl + "%") + "</div>" +
        '<div class="mc-chart-card"><h3>Monte Carlo results</h3><p>Distribution of final values across ' +
        fmtN(o.nSims) + " simulations. Light bars are losses, dark bars are profits (" + pct0(r.pProfit) + " profit).</p>" +
        histFinalSvg(r.rets, o.capital, money) +
        '<div class="mc-legend">' +
        '<span><i class="dash" style="border-color:#806d68"></i>Initial capital ' + formatRupiah(o.capital) + "</span>" +
        '<span><i style="border-color:#222222"></i>Median ' + formatRupiah(money.median) + "</span>" +
        '<span><i style="border-color:#e08a1e"></i>Mean ' + formatRupiah(money.mean) + "</span>" +
        '<span><i class="dash" style="border-color:#d62728"></i>P5 ' + formatRupiah(money.p5) + "</span>" +
        '<span><i class="dash" style="border-color:#2e8b3d"></i>P95 ' + formatRupiah(money.p95) + "</span></div>" +
        "</div></div>" +
                 '<div class="mc-chart-card"><h3>VaR vs Expected Shortfall</h3>' +
        "<p>Two different questions about the same worst " + tailPct + "% of scenarios.</p>" +
        '<div class="mc-risk">' +
        '<div><span>VALUE AT RISK (VaR ' + cl + "%)</span><strong>" + formatRupiah(varLoss * o.capital) + "</strong>" +
        "<small>The loss limit. In " + cl + "% of scenarios you lose no more than " + pct(varLoss) + " of your capital.</small></div>" +
        '<div class="es"><span>EXPECTED SHORTFALL (ES)</span><strong>' + formatRupiah(esLoss * o.capital) + "</strong>" +
        "<small>When losses do pass the VaR limit, the average loss is " + pct(esLoss) + ", which is " +
        formatRupiah(Math.max(0, esLoss - varLoss) * o.capital) + " deeper than VaR.</small></div></div>" +
        riskScaleSvg(varLoss, esLoss, worstLoss) +
        '<div class="mc-legend">' +
        '<span><i class="box" style="background:#c98a85"></i>Worst ' + tailPct + "% zone</span>" +
        '<span><i class="dash" style="border-color:#682127"></i>VaR</span>' +
        '<span><i style="border-color:#b4564f"></i>ES (average inside the zone)</span></div></div>' +
        '<div class="mc-warning"><h4>⚠ Watch out for Shortfall risk</h4><p>In the worst ' + tailPct +
        "% of scenarios, the average loss reaches " + pct(esLoss) + " (≈ " + formatRupiah(esLoss * o.capital) +
        "), deeper than the VaR limit of " + pct(varLoss) + " (≈ " + formatRupiah(varLoss * o.capital) +
        "). VaR only marks where losses begin to exceed the limit; it does not tell you how " +
        "severe they get beyond it. Expected Shortfall does.</p></div>" +

        '<div class="mc-res-two">' +
        '<div class="mc-read"><h4>How to read the results</h4>' +
        "<p><strong>" + pct0(r.pProfit) + " profit, " + pct0(lossP) + " loss.</strong> Out of " + fmtN(o.nSims) +
        " paths, about " + fmtN(nProfit) + " end above the initial capital. This is a probability, not a " +
        "guarantee: the other " + fmtN(nLoss) + " paths still end in a loss.</p>" +
        "<p><strong>VaR " + cl + "% " + pct(r.varR) + ".</strong> " + varText + "</p>" +
        "<p><strong>Shortfall " + pct(r.es) + ".</strong> The average loss when that worst-case zone actually happens.</p></div>" +
        '<div class="mc-read"><h4>What you should do</h4>' +
        "<p>" + advProfit + "</p>" +
        "<p><strong>VaR " + pct(r.varR) + ".</strong> Only allocate money you can afford to see drop by that much. " +
        "If that number makes you uncomfortable, reduce the position size.</p>" +
        "<p>" + diversify + "</p></div></div>" +

        '<p class="mc-disclaimer">This simulation uses historical data from ' + d.dates[0] + " to " +
        d.dates[d.dates.length - 1] + ". It is not investment advice and does not guarantee future results.</p>";

    simResults.innerHTML = html;
}


/* ---------- BOBOT MARKOWITZ (maksimum Sharpe) ---------- */
const RISK_FREE = 0.06, TRADING_DAYS = 252, MAX_WEIGHT = 0.4;
const avg = a => a.reduce((s, x) => s + x, 0) / a.length;

function covariance(R, means) {
    const n = R.length, T = R[0].length;
    const C = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
            let s = 0;
            for (let t = 0; t < T; t++) s += (R[i][t] - means[i]) * (R[j][t] - means[j]);
            C[i][j] = C[j][i] = s / (T - 1);
        }
    }
    return C;
}

// bobot 0 <= w <= cap dengan jumlah = 1
function projectWeights(v, cap) {
    let lo = Math.min(...v) - cap, hi = Math.max(...v);
    for (let k = 0; k < 60; k++) {
        const lam = (lo + hi) / 2;
        const sum = v.reduce((s, x) => s + Math.min(cap, Math.max(0, x - lam)), 0);
        if (sum > 1) lo = lam; else hi = lam;
    }
    const lam = (lo + hi) / 2;
    return v.map(x => Math.min(cap, Math.max(0, x - lam)));
}

function maxSharpeWeights(muA, covA, cap) {
    const n = muA.length;
    if (n === 1) return [1];
    let w = new Array(n).fill(1 / n), best = w.slice(), bestS = -Infinity;
    for (let k = 0; k < 3000; k++) {
        const cw = covA.map(row => row.reduce((s, c, j) => s + c * w[j], 0));
        const ret = w.reduce((s, x, i) => s + x * muA[i], 0);
        const risk = Math.sqrt(w.reduce((s, x, i) => s + x * cw[i], 0));
        const S = (ret - RISK_FREE) / risk;
        if (S > bestS) { bestS = S; best = w.slice(); }
        const lr = 0.3 / Math.sqrt(1 + k / 10);
        const g = muA.map((m, i) => m / risk - (ret - RISK_FREE) * cw[i] / Math.pow(risk, 3));
        w = projectWeights(w.map((x, i) => x + lr * g[i]), cap);
    }
    return best;
}

runButton.addEventListener("click", () => {
    const d = buildReturns(parseInt(document.getElementById("inPeriod").value, 10));
    if (d.dates.length < 61) {
        simResults.innerHTML = '<p class="mc-placeholder">Not enough data in the selected period. Try a longer period.</p>';
        return;
    }
    const o = {
        horizon: parseInt(document.getElementById("inHorizon").value, 10),
        nSims: parseInt(document.getElementById("inSims").value, 10),
        capital: parseFloat(document.getElementById("inCapital").value) || 100000000,
        conf: confLevel
    };

    runButton.disabled = true;
    runButton.textContent = "Running…";

    setTimeout(() => {
// return harian sederhana per saham (sama seperti pct_change di Python)
        const R = d.returns.map(row => row.map(x => Math.exp(x) - 1));
        const muD = R.map(avg);
        const covD = covariance(R, muD);
        const muA = muD.map(x => x * TRADING_DAYS);
        const covA = covD.map(row => row.map(c => c * TRADING_DAYS));
        const cap = Math.max(MAX_WEIGHT, 1 / R.length);
        const w = maxSharpeWeights(muA, covA, cap);

        d.cap = cap;
        d.stats = d.names.map((name, i) => {
            const vol = Math.sqrt(covA[i][i]);
            return { name, ret: muA[i], vol, sharpe: (muA[i] - RISK_FREE) / vol, w: w[i] };
        });
        d.daily = R[0].map((_, t) => R.reduce((s, row, i) => s + w[i] * row[t], 0));


        renderResults(runMonteCarlo(d.daily, o.nSims, o.horizon, o.conf), d, o);
        runButton.disabled = false;
        runButton.textContent = "Run Simulation";
        simResults.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 30);
});