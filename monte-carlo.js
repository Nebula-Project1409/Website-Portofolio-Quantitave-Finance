
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