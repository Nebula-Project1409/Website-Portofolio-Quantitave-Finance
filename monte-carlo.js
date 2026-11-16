
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
const KODE_URL = "../1_Model/monte_carlo.py";
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
            "Could not load monte_carlo.py. Open this page with Live Server " +
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
        document.getElementById("mSharpe").textContent = h.sharpe_portofolio.toFixed(2);
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

    for (let s = 0; s < nSims; s++) {
        let v = 1;
        const keep = s < 30 ? [1] : null;
        meanPath[0] += 1;
        for (let t = 1; t <= horizon; t++) {
            v *= 1 + mu + sd * randn();
            if (v < 0) v = 0;
            meanPath[t] += v;
            if (keep) keep.push(v);
        }
        finals[s] = v;
        if (keep) sample.push(keep);
    }
    for (let t = 0; t <= horizon; t++) meanPath[t] /= nSims;

    const rets = Array.from(finals, v => v - 1).sort((a, b) => a - b);
    const q = 1 - conf;
    const varR = pctile(rets, q);
    const tail = rets.filter(x => x <= varR);
    const es = tail.reduce((a, b) => a + b, 0) / tail.length;
    const pProfit = rets.filter(x => x > 0).length / nSims;

    const histSorted = daily.slice().sort((a, b) => a - b);
    return { rets, varR, es, pProfit, sample, meanPath, histVar: pctile(histSorted, q) };
}

function histSvg(values, split, label) {
    const bins = 14, W = 400, H = 170, pad = 18;
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

function pathsSvg(sample, meanPath) {
    const W = 800, H = 250, P = 10, T = meanPath.length - 1;
    let lo = 1, hi = 1;
    sample.forEach(p => p.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
    meanPath.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; });
    const X = t => P + t / T * (W - 2 * P);
    const Y = v => H - P - (v - lo) / ((hi - lo) || 1) * (H - 2 * P);
    const line = (arr, sw, col) =>
        '<polyline fill="none" stroke="' + col + '" stroke-width="' + sw + '" points="' +
        Array.from(arr, (v, t) => X(t).toFixed(1) + "," + Y(v).toFixed(1)).join(" ") + '"/>';
    return '<svg viewBox="0 0 ' + W + " " + H + '">' +
        sample.map(p => line(p, 1, "#d4a5a0")).join("") +
        '<line x1="' + P + '" x2="' + (W - P) + '" y1="' + Y(1).toFixed(1) + '" y2="' + Y(1).toFixed(1) +
        '" stroke="#806d68" stroke-dasharray="5 4"/>' +
        line(meanPath, 3, "#682127") + "</svg>";
}

function renderResults(r, d, o) {
    const cl = Math.round(o.conf * 100), tailPct = 100 - cl;
    const lossP = 1 - r.pProfit;
    const nProfit = Math.round(r.pProfit * o.nSims), nLoss = o.nSims - nProfit;
    const fmtN = x => x.toLocaleString("en-US");
    const varLoss = Math.max(0, -r.varR);

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
        : "<strong>Diversify wisely.</strong> This portfolio uses equal weights across " + d.names.length +
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
        "</strong><small>≈ " + rpShort(varLoss * o.capital) + " of " + rpShort(o.capital) + "</small></article>" +
        '<article class="mc-metric"><span>Expected Shortfall</span><strong>' + pct(r.es) +
        "</strong><small>average of the worst " + tailPct + "%</small></article></div>" +

        '<div class="mc-chart-card"><h3>Simulated price paths</h3><p>A sample of the ' + fmtN(o.nSims) +
        " paths. The thick line is the average; the dashed line is the initial capital.</p>" +
        pathsSvg(r.sample, r.meanPath) + "</div>" +

        '<div class="mc-res-two">' +
        '<div class="mc-chart-card"><h3>Historical VaR</h3><p>Distribution of daily returns from historical data.</p>' +
        histSvg(d.daily, r.histVar, "VaR " + cl + "%") + "</div>" +
        '<div class="mc-chart-card"><h3>Monte Carlo results</h3><p>Distribution of final values across ' +
        fmtN(o.nSims) + " simulations.</p>" + histSvg(r.rets, 0, "Initial capital") + "</div></div>" +

        '<div class="mc-warning"><h4>⚠ Watch out for Shortfall risk</h4><p>In the worst ' + tailPct +
        "% of scenarios, the average loss reaches " + pct(r.es) + ", deeper than the VaR limit of " +
        pct(r.varR) + ". VaR only marks where losses begin to exceed the limit; it does not tell you how " +
        "severe they get beyond it.</p></div>" +

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
        // return harian portofolio (bobot sama rata), dari log-return ke return sederhana
        const T = d.returns[0].length;
        d.daily = new Array(T);
        for (let t = 0; t < T; t++) {
            let s = 0;
            for (let i = 0; i < d.returns.length; i++) s += Math.exp(d.returns[i][t]) - 1;
            d.daily[t] = s / d.returns.length;
        }
        renderResults(runMonteCarlo(d.daily, o.nSims, o.horizon, o.conf), d, o);
        runButton.disabled = false;
        runButton.textContent = "Run Simulation";
        simResults.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 30);
});