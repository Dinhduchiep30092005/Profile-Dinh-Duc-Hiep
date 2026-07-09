/* ══════════════════════════════════════════
   GSC Dashboard — App Logic
   ══════════════════════════════════════════ */

// ── State ──
let currentReport = null;
let lastTitleGenData = null;  // Store SERP data for heading generation

// ── Navigation ──
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const page = document.getElementById('page-' + pageId);
    if (page) page.classList.add('active');

    const nav = document.querySelector(`.nav-item[data-page="${pageId}"]`);
    if (nav) nav.classList.add('active');
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// Init navigation
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        showPage(item.dataset.page);
        document.getElementById('sidebar').classList.remove('open');
    });
});

// ── Loading ──
function showLoading(text = 'Đang phân tích dữ liệu GSC...') {
    document.getElementById('loadingText').textContent = text;
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

// ── Toast ──
function toast(message, type = 'success') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3000);
}

// ── API Helper ──
async function api(endpoint, data = {}) {
    const res = await fetch('/api/' + endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || 'API error');
    return json;
}

// ── Format Helpers ──
function fmtNum(n) {
    if (n == null || n === '—') return '—';
    return Number(n).toLocaleString('vi-VN');
}

function fmtPct(n) {
    if (n == null) return '—';
    return (n * 100).toFixed(2) + '%';
}

function fmtPos(n) {
    if (n == null) return '—';
    return Number(n).toFixed(1);
}

function truncate(s, len = 40) {
    if (!s) return '';
    return s.length > len ? s.slice(0, len) + '…' : s;
}

function truncUrl(url, len = 35) {
    if (!url) return '';
    try {
        const u = new URL(url);
        const path = u.pathname;
        return path.length > len ? '…' + path.slice(-len + 3) : path;
    } catch {
        return truncate(url, len);
    }
}

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
}

// ── Main Analysis ──
async function runAnalysis() {
    const days = parseInt(document.getElementById('daysSelect').value);
    const btn = document.getElementById('btnAnalyze');
    btn.disabled = true;

    showLoading(`Đang phân tích ${days} ngày dữ liệu GSC...`);

    try {
        const report = await api('analyze', { days, top_n: 10 });
        currentReport = report;
        renderOverview(report);
        renderLowCtrTable(report.low_ctr_pages || []);
        renderGapsTable(report.impression_click_gaps || []);
        renderRewriteCards(report.title_rewrite_suggestions || []);
        toast(`Phân tích hoàn tất — ${report.summary.total_queries} queries`);
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        hideLoading();
        btn.disabled = false;
    }
}

// ── Render Overview ──
function renderOverview(report) {
    const s = report.summary || {};
    document.getElementById('statQueries').textContent = fmtNum(s.total_queries);
    document.getElementById('statClicks').textContent = fmtNum(s.total_clicks);
    document.getElementById('statImpressions').textContent = fmtNum(s.total_impressions);
    document.getElementById('statCTR').textContent = fmtPct(s.avg_ctr);
    document.getElementById('statPosition').textContent = fmtPos(s.avg_position);
    document.getElementById('statQuickWins').textContent = report.quick_wins || 0;

    // Overview low-CTR mini table
    const lcEntries = (report.low_ctr_pages || []).slice(0, 5);
    const lcTbody = document.querySelector('#overviewLowCtr tbody');
    if (lcEntries.length) {
        lcTbody.innerHTML = lcEntries.map(e => `
            <tr>
                <td title="${escHtml(e.query)}">${escHtml(truncate(e.query, 30))}</td>
                <td>${fmtPos(e.position)}</td>
                <td>${fmtNum(e.impressions)}</td>
                <td class="warning-text">${fmtPct(e.ctr)}</td>
                <td class="positive">+${Math.round(e.potential_clicks || 0)}</td>
            </tr>
        `).join('');
    } else {
        lcTbody.innerHTML = '<tr><td colspan="5" class="empty">Không có trang CTR thấp</td></tr>';
    }

    // Overview gaps mini table
    const gEntries = (report.impression_click_gaps || []).slice(0, 5);
    const gTbody = document.querySelector('#overviewGaps tbody');
    if (gEntries.length) {
        gTbody.innerHTML = gEntries.map(e => `
            <tr>
                <td title="${escHtml(e.query)}">${escHtml(truncate(e.query, 30))}</td>
                <td>${fmtPos(e.position)}</td>
                <td>${fmtNum(e.impressions)}</td>
                <td>${e.clicks}</td>
                <td class="negative">${Math.round(e.missed_clicks || 0)}</td>
            </tr>
        `).join('');
    } else {
        gTbody.innerHTML = '<tr><td colspan="5" class="empty">Không có gaps</td></tr>';
    }
}

// ── Render Low CTR Table ──
function renderLowCtrTable(entries) {
    const tbody = document.querySelector('#lowCtrTable tbody');
    if (!entries.length) {
        tbody.innerHTML = '<tr><td colspan="10" class="empty">Không có dữ liệu</td></tr>';
        return;
    }
    tbody.innerHTML = entries.map((e, i) => `
        <tr>
            <td>${i + 1}</td>
            <td title="${escHtml(e.query)}">${escHtml(truncate(e.query))}</td>
            <td title="${escHtml(e.page)}">${escHtml(truncUrl(e.page))}</td>
            <td>${fmtPos(e.position)}</td>
            <td>${fmtNum(e.impressions)}</td>
            <td>${e.clicks}</td>
            <td class="warning-text">${fmtPct(e.ctr)}</td>
            <td>${fmtPct(e.expected_ctr)}</td>
            <td class="negative">${fmtPct(e.ctr_gap)}</td>
            <td class="positive">+${Math.round(e.potential_clicks || 0)}</td>
        </tr>
    `).join('');
}

// ── Render Gaps Table ──
function renderGapsTable(entries) {
    const tbody = document.querySelector('#gapsTable tbody');
    if (!entries.length) {
        tbody.innerHTML = '<tr><td colspan="9" class="empty">Không có dữ liệu</td></tr>';
        return;
    }
    tbody.innerHTML = entries.map((e, i) => `
        <tr>
            <td>${i + 1}</td>
            <td title="${escHtml(e.query)}">${escHtml(truncate(e.query))}</td>
            <td title="${escHtml(e.page)}">${escHtml(truncUrl(e.page))}</td>
            <td>${fmtPos(e.position)}</td>
            <td>${fmtNum(e.impressions)}</td>
            <td>${e.clicks}</td>
            <td class="negative">${Math.round(e.waste_ratio || 0)}x</td>
            <td>${Math.round(e.expected_clicks || 0)}</td>
            <td class="negative">${Math.round(e.missed_clicks || 0)}</td>
        </tr>
    `).join('');
}

// ── Render Rewrite Cards ──
// (Legacy — kept for backward compat with full analysis)
function renderRewriteCards(entries) {
    // No-op: old rewrite rendering removed in favor of SERP title generator
}

// ══════════════════════════════════════════════
// SERP Title Generator
// ══════════════════════════════════════════════

function setPipelineStep(stepId, status) {
    // status: 'waiting' | 'active' | 'done' | 'error'
    const el = document.getElementById('step-' + stepId);
    if (!el) return;
    el.className = 'pipeline-step';
    const statusEl = el.querySelector('.step-status');
    switch (status) {
        case 'active':
            el.classList.add('step-active');
            statusEl.textContent = 'Đang xử lý...';
            break;
        case 'done':
            el.classList.add('step-done');
            statusEl.textContent = 'Hoàn tất ✓';
            break;
        case 'error':
            el.classList.add('step-error');
            statusEl.textContent = 'Lỗi ✗';
            break;
        default:
            statusEl.textContent = 'Đang chờ';
    }
}

function resetPipeline() {
    ['serp', 'extract', 'intent', 'gap', 'generate', 'ctr'].forEach(s => setPipelineStep(s, 'waiting'));
}

async function generateTitles() {
    const keyword = document.getElementById('rewriteKeyword').value.trim();
    if (!keyword) { toast('Vui lòng nhập từ khoá', 'error'); return; }

    const numTitles = parseInt(document.getElementById('rewriteNumTitles').value);
    const language = document.getElementById('rewriteLang').value;
    const btn = document.getElementById('btnGenerate');

    btn.disabled = true;
    document.getElementById('rewriteEmpty').style.display = 'none';
    document.getElementById('titleResults').style.display = 'none';
    document.getElementById('pipelineProgress').style.display = 'block';
    resetPipeline();

    // Animate pipeline steps
    const steps = ['serp', 'extract', 'intent', 'gap', 'generate', 'ctr'];
    let stepIndex = 0;
    const stepInterval = setInterval(() => {
        if (stepIndex < steps.length) {
            if (stepIndex > 0) setPipelineStep(steps[stepIndex - 1], 'done');
            setPipelineStep(steps[stepIndex], 'active');
            stepIndex++;
        }
    }, 2500);

    try {
        const res = await api('rewrite', { keyword, num_titles: numTitles, language });

        clearInterval(stepInterval);
        steps.forEach(s => setPipelineStep(s, 'done'));

        // Small delay to show all steps done
        await new Promise(r => setTimeout(r, 500));

        renderTitleGenResults(res);
        document.getElementById('titleResults').style.display = 'block';
        toast(`Tạo ${res.titles?.length || 0} title cho "${keyword}"`);
    } catch (err) {
        clearInterval(stepInterval);
        steps.forEach((s, i) => {
            if (i < stepIndex) setPipelineStep(s, i < stepIndex - 1 ? 'done' : 'error');
        });
        toast(err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

function renderTitleGenResults(data) {
    // Store data for heading generation
    lastTitleGenData = data;

    // Summary
    const summary = data.summary || {};
    const summaryEl = document.getElementById('titleGenSummary');
    const intentClass = {
        'informational': 'info', 'commercial': 'commercial',
        'comparison': 'comparison', 'transactional': 'transactional',
        'navigational': 'navigational'
    }[summary.dominant_intent] || 'info';

    summaryEl.innerHTML = `
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">${summary.total_serp || 0}</div>
                <div class="summary-label">SERP Results</div>
            </div>
            <div class="summary-item">
                <div class="summary-value intent-badge intent-${intentClass}">${summary.dominant_intent || '?'}</div>
                <div class="summary-label">Search Intent</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${Math.round((summary.confidence || 0) * 100)}%</div>
                <div class="summary-label">Confidence</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">${summary.gap_count || 0}</div>
                <div class="summary-label">Gaps Found</div>
            </div>
            <div class="summary-item">
                <div class="summary-value accent-text">${summary.titles_generated || 0}</div>
                <div class="summary-label">Titles Generated</div>
            </div>
        </div>
    `;

    // Intent Analysis
    const intent = data.intent || {};
    const intentEl = document.getElementById('intentResult');
    intentEl.innerHTML = `
        <div class="intent-analysis">
            <div class="intent-main">
                <span class="intent-badge intent-${intentClass}">${intent.intent_type || '?'}</span>
                <span class="intent-conf">Confidence: ${Math.round((intent.confidence || 0) * 100)}%</span>
            </div>
            ${intent.reasoning ? `<p class="intent-reason">${escHtml(intent.reasoning)}</p>` : ''}
            ${intent.user_persona ? `<p class="intent-persona"><strong>User:</strong> ${escHtml(intent.user_persona)}</p>` : ''}
            ${intent.content_expectation ? `<p class="intent-expect"><strong>Mong đợi:</strong> ${escHtml(intent.content_expectation)}</p>` : ''}
            ${intent.sub_intents?.length ? `<div class="sub-intents">Sub-intents: ${intent.sub_intents.map(s => `<span class="sub-badge">${escHtml(s)}</span>`).join(' ')}</div>` : ''}
        </div>
    `;

    // Gaps
    const gaps = data.gaps || {};
    const gapEl = document.getElementById('gapResult');
    const intentGaps = gaps.intent_gaps || [];
    gapEl.innerHTML = `
        ${intentGaps.length ? intentGaps.map(g => `
            <div class="gap-item">
                <div class="gap-header">
                    <span class="gap-score">${g.opportunity_score || '?'}/10</span>
                    <span class="gap-text">${escHtml(g.gap)}</span>
                </div>
                ${g.reason ? `<p class="gap-reason">${escHtml(g.reason)}</p>` : ''}
                ${g.suggested_approach ? `<p class="gap-approach"><em>${escHtml(g.suggested_approach)}</em></p>` : ''}
            </div>
        `).join('') : '<p class="empty">Không phát hiện gap đáng kể</p>'}
        ${gaps.differentiation_tips?.length ? `
            <div class="diff-tips">
                <strong>Tips khác biệt:</strong>
                <ul>${gaps.differentiation_tips.map(t => `<li>${escHtml(t)}</li>`).join('')}</ul>
            </div>
        ` : ''}
    `;

    document.getElementById('serpAnalysisGrid').style.display = 'grid';

    // SERP Table
    const serp = data.serp_results || [];
    if (serp.length) {
        const tbody = document.querySelector('#serpTable tbody');
        tbody.innerHTML = serp.map(r => `
            <tr>
                <td>${r.position}</td>
                <td title="${escHtml(r.title)}">
                    <a href="${escHtml(r.url)}" target="_blank" rel="noopener noreferrer" class="serp-title-link">${escHtml(truncate(r.title, 55))}</a>
                </td>
                <td class="domain-cell">${escHtml(r.domain)}</td>
                <td>${r.title_length}</td>
                <td>${r.has_number ? '✓' : '—'}</td>
                <td>${r.has_year ? '✓' : '—'}</td>
                <td>${r.has_power_word ? '✓' : '—'}</td>
            </tr>
        `).join('');
        document.getElementById('serpTableCard').style.display = 'block';
    }

    // Generated Titles
    const titles = data.titles || [];
    const titlesContainer = document.getElementById('generatedTitles');
    if (!titles.length) {
        titlesContainer.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠️</div><p>Không tạo được title — thử lại hoặc thay đổi từ khoá</p></div>`;
        return;
    }

    titlesContainer.innerHTML = `<h3 style="margin: 1.5rem 0 1rem; color: #e0e0e0;">🏆 Title Suggestions (sorted by predicted CTR) — <span style="color: var(--accent); font-size: 0.85rem;">💡 Click vào title để tạo cấu trúc heading</span></h3>` +
        titles.map((t, i) => {
            const ctrPct = ((t.predicted_ctr || 0) * 100).toFixed(1);
            const ctrColor = t.predicted_ctr >= 0.08 ? '#4caf50' : t.predicted_ctr >= 0.05 ? '#ff9800' : '#f44336';
            const lenColor = t.title_length <= 60 ? '#4caf50' : '#ff9800';
            return `
            <div class="rewrite-card title-gen-card">
                <div class="rewrite-card-header pri-header-${i === 0 ? 'high' : i < 3 ? 'medium' : 'low'}">
                    <span>#${t.rank || i + 1} — ${escHtml(t.angle || 'Title')}</span>
                    <span class="ctr-badge" style="background:${ctrColor}">CTR: ${ctrPct}%</span>
                </div>
                <div class="rewrite-card-body">
                    <div class="rewrite-field full-row rewrite-new title-clickable" onclick="generateHeadingsForTitle(${i})" title="Click để tạo cấu trúc heading SEO + EEAT">
                        <label>✏️ Title <span class="heading-gen-hint">— click để tạo heading structure 📐</span></label>
                        <div class="value title-main" style="font-weight:700; font-size:1.1rem; cursor:pointer;">${escHtml(t.title || '—')}</div>
                        <div class="title-meta">
                            <span class="len-badge" style="color:${lenColor}">${t.title_length} ký tự</span>
                            ${(t.power_elements || []).map(p => `<span class="power-badge">${escHtml(p)}</span>`).join('')}
                        </div>
                    </div>
                    <div class="rewrite-field full-row rewrite-new">
                        <label>📝 Meta Description</label>
                        <div class="value">${escHtml(t.meta_description || '—')}</div>
                    </div>
                    <div class="rewrite-field">
                        <label>🎯 Target Intent</label>
                        <div class="value"><span class="intent-badge intent-${(t.target_intent || '').replace(' ', '-')}">${escHtml(t.target_intent || '—')}</span></div>
                    </div>
                    <div class="rewrite-field">
                        <label>📈 CTR Prediction</label>
                        <div class="value">
                            <div class="ctr-bar">
                                <div class="ctr-fill" style="width:${Math.min(ctrPct * 3, 100)}%; background:${ctrColor};"></div>
                            </div>
                            <span style="color:${ctrColor}; font-weight:600;">${ctrPct}%</span>
                        </div>
                    </div>
                    <div class="rewrite-field full-row">
                        <label>💡 CTR Analysis</label>
                        <div class="value">${escHtml(t.ctr_reasoning || '—')}</div>
                    </div>
                    <div class="rewrite-field full-row">
                        <label>🔄 Differentiation</label>
                        <div class="value">${escHtml(t.differentiation || '—')}</div>
                    </div>
                </div>
            </div>`;
        }).join('');
}

function toggleSerpTable() {
    const body = document.getElementById('serpTableBody');
    body.style.display = body.style.display === 'none' ? 'block' : 'none';
}

// ── Load Low CTR (standalone) ──
async function loadLowCtr() {
    const days = parseInt(document.getElementById('daysSelect').value);
    const ctr = parseFloat(document.getElementById('ctrThreshold').value);
    const minImpr = parseInt(document.getElementById('minImpressions').value);

    showLoading('Đang tìm trang CTR thấp...');
    try {
        const res = await api('low-ctr', { days, ctr_threshold: ctr, min_impressions: minImpr });
        renderLowCtrTable(res.entries || []);
        toast(`Tìm thấy ${res.total} trang CTR thấp`);
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ── Load Gaps (standalone) ──
async function loadGaps() {
    const days = parseInt(document.getElementById('daysSelect').value);
    const minImpr = parseInt(document.getElementById('gapMinImpr').value);
    const maxClicks = parseInt(document.getElementById('gapMaxClicks').value);

    showLoading('Đang tìm impression gaps...');
    try {
        const res = await api('gaps', { days, min_impressions: minImpr, max_clicks: maxClicks });
        renderGapsTable(res.entries || []);
        toast(`Tìm thấy ${res.total} gaps`);
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ── Load Rewrites (standalone) — now uses SERP Title Generator ──
async function loadRewrites() {
    // Redirect to generateTitles for backward compatibility
    generateTitles();
}

// ── Page Analysis ──
async function analyzePage() {
    const url = document.getElementById('pageUrl').value.trim();
    if (!url) { toast('Vui lòng nhập URL', 'error'); return; }

    const days = parseInt(document.getElementById('daysSelect').value);
    const container = document.getElementById('pageResult');

    showLoading(`Đang phân tích ${url}...`);
    try {
        const res = await api('page', { url, days });

        if (!res.queries || !res.queries.length) {
            container.innerHTML = `<div class="empty-state"><div class="empty-icon">🤷</div><p>Không có dữ liệu cho URL này</p></div>`;
            return;
        }

        const queries = res.queries.sort((a, b) => b.impressions - a.impressions);

        container.innerHTML = `
            <div class="page-stats">
                <div class="page-stat"><div class="value">${res.total_queries}</div><div class="label">Queries</div></div>
                <div class="page-stat"><div class="value">${fmtNum(res.total_clicks)}</div><div class="label">Clicks</div></div>
                <div class="page-stat"><div class="value">${fmtNum(res.total_impressions)}</div><div class="label">Impressions</div></div>
                <div class="page-stat"><div class="value">${fmtPct(res.avg_ctr)}</div><div class="label">Avg CTR</div></div>
            </div>
            <div class="card">
                <div class="card-header"><h3>📋 Queries cho trang này</h3></div>
                <div class="card-body">
                    <table class="data-table full-width">
                        <thead><tr><th>#</th><th>Query</th><th>Pos</th><th>Impr</th><th>Clicks</th><th>CTR</th></tr></thead>
                        <tbody>
                            ${queries.slice(0, 50).map((q, i) => {
                                const ctrClass = q.ctr < 0.03 ? 'negative' : (q.ctr < 0.05 ? 'warning-text' : 'positive');
                                return `<tr>
                                    <td>${i + 1}</td>
                                    <td title="${escHtml(q.query)}">${escHtml(truncate(q.query, 50))}</td>
                                    <td>${fmtPos(q.position)}</td>
                                    <td>${fmtNum(q.impressions)}</td>
                                    <td>${q.clicks}</td>
                                    <td class="${ctrClass}">${fmtPct(q.ctr)}</td>
                                </tr>`;
                            }).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
        toast(`Phân tích ${res.total_queries} queries cho trang`);
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        hideLoading();
    }
}

// ── Save Settings ──
async function saveSettings() {
    const creds = document.getElementById('settingsCreds').value.trim();
    const url = document.getElementById('settingsUrl').value.trim();
    const openaiKey = document.getElementById('settingsOpenaiKey').value.trim();
    const openaiModel = document.getElementById('settingsOpenaiModel').value.trim();
    const serpApiKey = document.getElementById('settingsSerpApiKey').value.trim();
    const msgEl = document.getElementById('settingsMsg');

    if (!creds || !url) {
        msgEl.innerHTML = '<span class="negative">Vui lòng nhập đầy đủ thông tin GSC</span>';
        return;
    }

    try {
        const res = await api('save-settings', {
            credentials_path: creds,
            site_url: url,
            openai_api_key: openaiKey,
            openai_model: openaiModel,
            serpapi_key: serpApiKey,
        });
        msgEl.innerHTML = '<span class="positive">✓ Đã lưu cài đặt thành công!</span>';
        document.getElementById('siteUrl').textContent = url;
        toast('Cài đặt đã lưu');
    } catch (err) {
        msgEl.innerHTML = `<span class="negative">✗ Lỗi: ${escHtml(err.message)}</span>`;
    }
}

// ══════════════════════════════════════════════
// Heading Structure Generator
// ══════════════════════════════════════════════

let lastHeadingResult = null;
let manualEntryHeadingContext = null; // Stores { keyword, title, language } when generated from manual entry modal
let headingModalFromManual = false;   // True when heading modal was opened from manual entry modal

async function generateHeadingsForManualEntry() {
    const keyword = document.getElementById('manualKeyword').value.trim();
    const title = document.getElementById('manualTitle').value.trim();
    if (!keyword) { toast('Vui lòng nhập keyword trước.', 'error'); return; }
    if (!title) { toast('Vui lòng nhập title trước.', 'error'); return; }

    const language = document.getElementById('manualLang').value;
    manualEntryHeadingContext = { keyword, title, language };
    headingModalFromManual = true;

    // Show heading modal footer buttons appropriately
    document.getElementById('btnSaveToProject').style.display = 'none';
    document.getElementById('btnUseHeadingManual').style.display = '';

    // Hide manual entry modal so heading modal appears on top
    document.getElementById('manualEntryModal').style.display = 'none';

    openHeadingModal(title);

    try {
        const res = await api('generate-headings', {
            title,
            keyword,
            language,
            intent: {},
            gaps: {},
            serp_results: [],
            people_also_ask: [],
        });

        lastHeadingResult = res;
        renderHeadingStructure(res, title, keyword);
        toast(`Đã tạo cấu trúc heading cho "${truncate(title, 40)}"`);
    } catch (err) {
        document.getElementById('headingModalBody').innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">❌</div>
                <p>Lỗi khi tạo heading: ${escHtml(err.message)}</p>
            </div>
        `;
        toast(err.message, 'error');
    }
}

function openHeadingModalFromManual() {
    // Re-open the heading modal to view the last generated result
    if (!lastHeadingResult || !manualEntryHeadingContext) {
        toast('Chưa có cấu trúc heading.', 'error');
        return;
    }
    headingModalFromManual = true;
    document.getElementById('btnSaveToProject').style.display = 'none';
    document.getElementById('btnUseHeadingManual').style.display = '';
    // Hide manual entry modal so heading modal appears on top
    document.getElementById('manualEntryModal').style.display = 'none';
    document.getElementById('headingModalTitle').textContent = `📐 Heading Structure: ${truncate(manualEntryHeadingContext.title, 50)}`;
    renderHeadingStructure(lastHeadingResult, manualEntryHeadingContext.title, manualEntryHeadingContext.keyword);
    document.getElementById('headingModal').style.display = 'flex';
}

function useHeadingForManualEntry() {
    // Close heading modal and return to manual entry, marking heading as ready
    closeHeadingModal();
    document.getElementById('manualHeadingPreview').style.display = 'block';
}

async function generateHeadingsForTitle(titleIndex) {
    if (!lastTitleGenData || !lastTitleGenData.titles) {
        toast('Chưa có dữ liệu title. Hãy tạo title trước.', 'error');
        return;
    }

    const titleObj = lastTitleGenData.titles[titleIndex];
    if (!titleObj) { toast('Title không hợp lệ', 'error'); return; }

    const title = titleObj.title;
    const keyword = lastTitleGenData.keyword;
    const language = document.getElementById('rewriteLang')?.value || 'vi';

    // Open modal with loading state (Title Rewrites context — restore default buttons)
    headingModalFromManual = false;
    _resetHeadingModalButtons();
    openHeadingModal(title);

    try {
        const res = await api('generate-headings', {
            title,
            keyword,
            language,
            intent: lastTitleGenData.intent || {},
            gaps: lastTitleGenData.gaps || {},
            serp_results: lastTitleGenData.serp_results || [],
            people_also_ask: lastTitleGenData.people_also_ask || [],
        });

        lastHeadingResult = res;
        renderHeadingStructure(res, title, keyword);
        toast(`Đã tạo cấu trúc heading cho "${truncate(title, 40)}"`);
    } catch (err) {
        document.getElementById('headingModalBody').innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">❌</div>
                <p>Lỗi khi tạo heading: ${escHtml(err.message)}</p>
            </div>
        `;
        toast(err.message, 'error');
    }
}

function _resetHeadingModalButtons() {
    // Restore default heading modal footer buttons (for Title Rewrites context)
    const btnSave = document.getElementById('btnSaveToProject');
    const btnUse = document.getElementById('btnUseHeadingManual');
    if (btnSave) btnSave.style.display = '';
    if (btnUse) btnUse.style.display = 'none';
}

function openHeadingModal(title) {
    document.getElementById('headingModalTitle').textContent = `📐 Heading Structure: ${truncate(title, 50)}`;
    document.getElementById('headingModalBody').innerHTML = `
        <div class="rag-pipeline-loading">
            <div class="rag-pipeline-steps">
                <div class="rag-step active" id="ragStep1">
                    <div class="rag-step-icon">🎯</div>
                    <div class="rag-step-label">Phân tích Intent</div>
                </div>
                <div class="rag-step-arrow">→</div>
                <div class="rag-step" id="ragStep2">
                    <div class="rag-step-icon">🔍</div>
                    <div class="rag-step-label">Semantic Search</div>
                </div>
                <div class="rag-step-arrow">→</div>
                <div class="rag-step" id="ragStep3">
                    <div class="rag-step-icon">📑</div>
                    <div class="rag-step-label">Extract Chunks</div>
                </div>
                <div class="rag-step-arrow">→</div>
                <div class="rag-step" id="ragStep4">
                    <div class="rag-step-icon">💉</div>
                    <div class="rag-step-label">Inject Prompt</div>
                </div>
                <div class="rag-step-arrow">→</div>
                <div class="rag-step" id="ragStep5">
                    <div class="rag-step-icon">✨</div>
                    <div class="rag-step-label">Generate Headings</div>
                </div>
            </div>
            <div class="spinner" style="margin-top:1.5rem;"></div>
            <p style="margin-top:1rem; color: var(--text-secondary);">Đang chạy RAG Pipeline...</p>
            <p style="color: var(--text-dim); font-size:0.85rem;">Có thể mất 10-20 giây</p>
        </div>
    `;
    document.getElementById('headingModal').style.display = 'flex';

    // Animate pipeline steps
    const steps = ['ragStep1','ragStep2','ragStep3','ragStep4','ragStep5'];
    steps.forEach((id, i) => {
        setTimeout(() => {
            const el = document.getElementById(id);
            if (el) { el.classList.add('active'); el.classList.add('pulse'); }
        }, i * 2500);
    });
}

function closeHeadingModal() {
    document.getElementById('headingModal').style.display = 'none';
    if (headingModalFromManual) {
        // Restore manual entry modal
        document.getElementById('manualEntryModal').style.display = 'flex';
    } else {
        _resetHeadingModalButtons();
    }
    headingModalFromManual = false;
}

function renderHeadingStructure(data, title, keyword) {
    const body = document.getElementById('headingModalBody');

    const sections = data.sections || [];
    const faq = data.faq || [];
    const eeat = data.eeat_summary || {};
    const meta = data.meta || {};
    const intro = data.intro || {};

    // Build heading tree HTML
    let headingTree = '';

    // H1
    headingTree += `
        <div class="heading-node heading-h1">
            <span class="heading-tag">H1</span>
            <span class="heading-text">${escHtml(data.h1 || title)}</span>
        </div>
    `;

    // Intro
    if (intro.purpose) {
        headingTree += `
            <div class="heading-node heading-intro">
                <span class="heading-tag tag-intro">INTRO</span>
                <span class="heading-text">${escHtml(intro.purpose)}</span>
                <span class="eeat-badge eeat-${intro.eeat_signal || 'experience'}">${intro.eeat_signal || 'experience'}</span>
            </div>
        `;
    }

    // Sections
    for (const sec of sections) {
        const isH2 = sec.level === 'h2';
        const isH3 = sec.level === 'h3';
        const isH4 = sec.level === 'h4';
        const nestClass = isH2 ? '' : (isH4 ? 'heading-nested heading-deep' : 'heading-nested');
        const eatClass = sec.eeat_signal || 'expertise';
        const brandBadge = sec.brand_knowledge_used
            ? '<span class="brand-badge">📚 Brand Knowledge</span>'
            : '';
        const contentTypeBadge = sec.content_type
            ? `<span class="content-type-badge">${escHtml(sec.content_type)}</span>`
            : '';
        const purposeHtml = sec.purpose
            ? `<div class="heading-purpose ${isH2 ? '' : (isH4 ? 'heading-purpose-deep' : 'heading-purpose-nested')}">${escHtml(sec.purpose)}</div>`
            : '';

        headingTree += `
            <div class="heading-node heading-${sec.level} ${nestClass}">
                <span class="heading-tag tag-${sec.level}">${sec.level.toUpperCase()}</span>
                <span class="heading-text">${escHtml(sec.heading)}</span>
                <span class="eeat-badge eeat-${eatClass}">${eatClass}</span>
                ${contentTypeBadge}
                ${brandBadge}
                ${sec.word_count_target ? `<span class="wc-badge">${sec.word_count_target}w</span>` : ''}
            </div>
            ${purposeHtml}
            ${sec.key_points?.length ? `
                <div class="heading-key-points ${isH2 ? '' : (isH4 ? 'heading-deep-points' : 'heading-nested-points')}">
                    ${sec.key_points.map(p => `<span class="key-point">• ${escHtml(p)}</span>`).join('')}
                </div>
            ` : ''}
        `;
    }

    // FAQ section
    let faqHtml = '';
    if (faq.length) {
        faqHtml = `
            <div class="heading-section">
                <h4>❓ FAQ Schema-Ready (${faq.length} câu hỏi)</h4>
                <div class="faq-list">
                    ${faq.map(f => `
                        <div class="faq-item">
                            <div class="faq-q">
                                <span class="faq-icon">Q</span>
                                ${escHtml(f.question)}
                                <span class="faq-source faq-${f.source || 'generated'}">${f.source || 'generated'}</span>
                            </div>
                            ${f.answer_outline ? `<div class="faq-a">${escHtml(f.answer_outline)}</div>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // EEAT Summary
    let eeatHtml = '';
    if (eeat.experience_sections || eeat.expertise_sections) {
        eeatHtml = `
            <div class="heading-section">
                <h4>🏆 EEAT Coverage Map</h4>
                <div class="eeat-map">
                    ${_renderEeatMap('Experience', eeat.experience_sections)}
                    ${_renderEeatMap('Expertise', eeat.expertise_sections)}
                    ${_renderEeatMap('Authority', eeat.authority_sections)}
                    ${_renderEeatMap('Trust', eeat.trust_sections)}
                </div>
                ${eeat.brand_knowledge_impact ? `
                    <div class="brand-impact">
                        <strong>📚 Brand Knowledge Impact:</strong> ${escHtml(eeat.brand_knowledge_impact)}
                    </div>
                ` : ''}
            </div>
        `;
    }

    // Meta info
    let metaHtml = '';
    if (meta.total_sections || meta.estimated_word_count) {
        metaHtml = `
            <div class="heading-meta-bar">
                ${meta.total_sections ? `<div class="meta-item"><strong>${meta.total_sections}</strong> sections</div>` : ''}
                ${meta.h2_count ? `<div class="meta-item"><span class="heading-tag tag-h2" style="font-size:0.65rem">H2</span> <strong>${meta.h2_count}</strong></div>` : ''}
                ${meta.h3_count ? `<div class="meta-item"><span class="heading-tag tag-h3" style="font-size:0.65rem">H3</span> <strong>${meta.h3_count}</strong></div>` : ''}
                ${meta.h4_count ? `<div class="meta-item"><span class="heading-tag tag-h4" style="font-size:0.65rem">H4</span> <strong>${meta.h4_count}</strong></div>` : ''}
                ${meta.estimated_word_count ? `<div class="meta-item"><strong>${meta.estimated_word_count.toLocaleString()}</strong> words</div>` : ''}
                ${meta.estimated_read_time ? `<div class="meta-item"><strong>${meta.estimated_read_time}</strong> min read</div>` : ''}
                ${meta.content_depth_score ? `<div class="meta-item">Depth: <strong>${meta.content_depth_score}/10</strong></div>` : ''}
            </div>
        `;
    }

    body.innerHTML = `
        ${metaHtml}
        ${_renderRagInfoPanel(data.rag_info)}
        <div class="heading-tree">${headingTree}</div>
        ${faqHtml}
        ${eeatHtml}
    `;
}

function _renderRagInfoPanel(ragInfo) {
    if (!ragInfo) return '';
    const intentColors = {
        informational: '#38bdf8', commercial: '#fbbf24',
        comparison: '#a855f7', transactional: '#4ade80'
    };
    const intentColor = intentColors[ragInfo.intent_type] || '#94a3b8';

    let html = `
        <div class="rag-info-panel">
            <div class="rag-info-header">
                <span class="rag-info-title">🔗 RAG Pipeline</span>
                <span class="rag-intent-badge" style="background:${intentColor}20; color:${intentColor}; border:1px solid ${intentColor}40;">
                    ${ragInfo.intent_type || 'unknown'}
                </span>
            </div>
    `;

    if (ragInfo.has_knowledge) {
        html += `
            <div class="rag-stats">
                <div class="rag-stat">
                    <span class="rag-stat-value">${ragInfo.queries_executed || 0}</span>
                    <span class="rag-stat-label">Queries</span>
                </div>
                <div class="rag-stat">
                    <span class="rag-stat-value">${ragInfo.chunks_found || 0}</span>
                    <span class="rag-stat-label">Chunks</span>
                </div>
                <div class="rag-stat">
                    <span class="rag-stat-value">${ragInfo.categories_found || 0}</span>
                    <span class="rag-stat-label">Categories</span>
                </div>
                <div class="rag-stat">
                    <span class="rag-stat-value">${ragInfo.documents_used || 0}</span>
                    <span class="rag-stat-label">Documents</span>
                </div>
            </div>
        `;

        if (ragInfo.categories?.length) {
            html += `<div class="rag-categories">`;
            for (const cat of ragInfo.categories) {
                html += `<span class="rag-cat-tag">${escHtml(cat)}</span>`;
            }
            html += `</div>`;
        }

        if (ragInfo.queries_used?.length) {
            html += `
                <details class="rag-queries-detail">
                    <summary>Xem ${ragInfo.queries_used.length} search queries</summary>
                    <div class="rag-queries-list">
                        ${ragInfo.queries_used.map(q => `<code class="rag-query-item">${escHtml(q)}</code>`).join('')}
                    </div>
                </details>
            `;
        }
    } else {
        html += `
            <div class="rag-no-knowledge">
                <span>📭</span> Chưa upload tài liệu. <a href="#" onclick="showPage('knowledge');closeHeadingModal();return false;">Upload ngay</a>
            </div>
        `;
    }

    html += `</div>`;
    return html;
}

function _renderEeatMap(label, sections) {
    if (!sections || !sections.length) return '';
    const colorMap = {
        'Experience': '#4ade80', 'Expertise': '#38bdf8',
        'Authority': '#a855f7', 'Trust': '#fbbf24'
    };
    const color = colorMap[label] || '#94a3b8';
    return `
        <div class="eeat-map-row">
            <div class="eeat-map-label" style="color:${color};">${label}</div>
            <div class="eeat-map-items">
                ${sections.map(s => `<span class="eeat-map-item">${escHtml(typeof s === 'string' ? s : s.heading || s)}</span>`).join('')}
            </div>
        </div>
    `;
}

function copyHeadingStructure() {
    if (!lastHeadingResult) { toast('Chưa có heading structure', 'error'); return; }

    let text = '';
    const data = lastHeadingResult;

    // H1
    text += `# ${data.h1 || ''}\n\n`;

    // Intro
    if (data.intro?.purpose) {
        text += `[Intro: ${data.intro.purpose}]\n\n`;
    }

    // Sections
    for (const sec of (data.sections || [])) {
        const prefix = sec.level === 'h2' ? '## ' : '### ';
        text += `${prefix}${sec.heading}\n`;
        if (sec.key_points?.length) {
            for (const p of sec.key_points) {
                text += `  - ${p}\n`;
            }
        }
        text += '\n';
    }

    // FAQ
    if (data.faq?.length) {
        text += '## FAQ\n\n';
        for (const f of data.faq) {
            text += `**Q: ${f.question}**\n`;
            if (f.answer_outline) text += `A: ${f.answer_outline}\n`;
            text += '\n';
        }
    }

    navigator.clipboard.writeText(text).then(() => {
        toast('Đã copy heading structure!');
    }).catch(() => {
        // Fallback
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        ta.remove();
        toast('Đã copy heading structure!');
    });
}

function exportHeadingMarkdown() {
    if (!lastHeadingResult) { toast('Chưa có heading structure', 'error'); return; }

    let md = '';
    const data = lastHeadingResult;

    md += `# ${data.h1 || ''}\n\n`;

    if (data.intro?.purpose) {
        md += `> **Intro:** ${data.intro.purpose}\n\n`;
    }

    for (const sec of (data.sections || [])) {
        const prefix = sec.level === 'h2' ? '## ' : '### ';
        md += `${prefix}${sec.heading}`;
        if (sec.eeat_signal) md += ` <!-- EEAT: ${sec.eeat_signal} -->`;
        md += '\n\n';
        if (sec.purpose) md += `> ${sec.purpose}\n\n`;
        if (sec.key_points?.length) {
            for (const p of sec.key_points) md += `- ${p}\n`;
            md += '\n';
        }
    }

    if (data.faq?.length) {
        md += '## FAQ (Schema-Ready)\n\n';
        for (const f of data.faq) {
            md += `### ${f.question}\n\n`;
            if (f.answer_outline) md += `${f.answer_outline}\n\n`;
        }
    }

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `heading-structure-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast('Đã export Markdown file!');
}

// Close modal on backdrop click
document.addEventListener('click', (e) => {
    if (e.target.id === 'headingModal') closeHeadingModal();
});

// ══════════════════════════════════════════════
// Knowledge Base — Document Upload
// ══════════════════════════════════════════════

// File upload via drag & drop and click
document.addEventListener('DOMContentLoaded', () => {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    if (zone) {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('drag-over');
        });
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) uploadDocument(e.dataTransfer.files[0]);
        });
    }

    if (input) {
        input.addEventListener('change', () => {
            if (input.files.length) uploadDocument(input.files[0]);
            input.value = '';
        });
    }
});

async function uploadDocument(file) {
    const maxSize = 30 * 1024 * 1024;
    if (file.size > maxSize) {
        toast('File quá lớn. Tối đa 30MB.', 'error');
        return;
    }

    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const allowed = ['.pdf', '.docx', '.txt', '.md', '.markdown'];
    if (!allowed.includes(ext)) {
        toast(`Định dạng không hỗ trợ: ${ext}`, 'error');
        return;
    }

    const progress = document.getElementById('uploadProgress');
    const progressText = document.getElementById('uploadProgressText');
    const progressFill = document.getElementById('uploadProgressFill');

    progress.style.display = 'block';
    progressText.textContent = `Đang upload "${file.name}"...`;
    progressFill.style.width = '30%';

    const formData = new FormData();
    formData.append('file', file);

    try {
        progressText.textContent = 'Đang trích xuất nội dung & tạo embeddings...';
        progressFill.style.width = '60%';

        const res = await fetch('/api/upload-document', {
            method: 'POST',
            body: formData,
        });
        const json = await res.json();

        if (!res.ok) throw new Error(json.error || 'Upload failed');

        progressFill.style.width = '100%';

        // Handle warning (file saved but embedding failed, e.g. quota exceeded)
        if (json.warning) {
            progressText.textContent = `⚠ ${json.message}`;
            progressFill.style.background = '#fbbf24';
            toast(json.warning, 'error');
        } else {
            progressText.textContent = `✓ ${json.message}`;
            toast(json.message);
        }

        loadDocuments();

        setTimeout(() => {
            progress.style.display = 'none';
            progressFill.style.width = '0';
            progressFill.style.background = '';
        }, 3000);
    } catch (err) {
        progressText.textContent = `✗ Lỗi: ${err.message}`;
        progressFill.style.width = '100%';
        progressFill.style.background = 'var(--danger)';
        toast(err.message, 'error');

        setTimeout(() => {
            progress.style.display = 'none';
            progressFill.style.width = '0';
            progressFill.style.background = '';
        }, 3000);
    }
}

async function loadDocuments() {
    const container = document.getElementById('documentList');
    if (!container) return;

    try {
        const res = await fetch('/api/documents');
        const json = await res.json();

        if (!json.documents || !json.documents.length) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📚</div>
                    <p>Chưa có tài liệu nào. Upload tài liệu để bắt đầu.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = json.documents.map(doc => {
            const extIcon = { '.pdf': '📕', '.docx': '📘', '.txt': '📝', '.md': '📄', '.markdown': '📄' }[doc.extension] || '📄';
            const date = new Date(doc.upload_time).toLocaleDateString('vi-VN', {
                day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'
            });
            const embeddingStatus = doc.has_embeddings
                ? '<span class="positive">✓ Có embeddings</span>'
                : '<span class="warning-text">⚠ Chưa embed</span>';

            return `
                <div class="doc-item">
                    <div class="doc-icon">${extIcon}</div>
                    <div class="doc-info">
                        <div class="doc-name">${escHtml(doc.name)}</div>
                        <div class="doc-meta">
                            ${date} · ${(doc.text_length / 1000).toFixed(1)}K ký tự · ${doc.chunk_count} chunks · ${embeddingStatus}
                        </div>
                    </div>
                    <div class="doc-actions">
                        ${!doc.has_embeddings ? `<button class="btn btn-sm" onclick="reembedDoc('${doc.id}')">🔄 Embed</button>` : ''}
                        <button class="btn btn-sm btn-danger-sm" onclick="deleteDoc('${doc.id}', '${escHtml(doc.name)}')">🗑</button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = `<p class="negative">Lỗi tải danh sách: ${escHtml(err.message)}</p>`;
    }
}

async function deleteDoc(docId, docName) {
    if (!confirm(`Xoá tài liệu "${docName}"?`)) return;

    try {
        const res = await fetch(`/api/delete-document/${docId}`, { method: 'DELETE' });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast('Đã xoá tài liệu');
        loadDocuments();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function reembedDoc(docId) {
    try {
        toast('Đang tạo embeddings...');
        const res = await fetch(`/api/reembed-document/${docId}`, { method: 'POST' });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast(`Đã tạo ${json.embeddings_created} embeddings`);
        loadDocuments();
    } catch (err) {
        toast(err.message, 'error');
    }
}

// Auto-load documents when knowledge page is shown
document.addEventListener('DOMContentLoaded', () => {
    const knowledgeNav = document.querySelector('.nav-item[data-page="knowledge"]');
    if (knowledgeNav) {
        knowledgeNav.addEventListener('click', () => loadDocuments());
    }
});

// ── Keyboard Shortcuts ──
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        runAnalysis();
    }
});

// Enter key on keyword input triggers title generation
document.addEventListener('DOMContentLoaded', () => {
    const kwInput = document.getElementById('rewriteKeyword');
    if (kwInput) {
        kwInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                generateTitles();
            }
        });
    }
});

// ══════════════════════════════════════════════
// Project Management
// ══════════════════════════════════════════════

let currentProjectId = null;
let currentProjectName = '';
let saveKeywordCache = '';
let saveTitleCache = '';
let saveHeadingCache = null;

// ── Load & Render Projects ──

async function loadProjects() {
    const container = document.getElementById('projectList');
    if (!container) return;

    try {
        const res = await fetch('/api/projects');
        const json = await res.json();

        if (!json.projects || !json.projects.length) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📁</div>
                    <p>Chưa có project nào. Tạo project đầu tiên ở trên.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = json.projects.map(p => {
            const date = new Date(p.created_at).toLocaleDateString('vi-VN', {
                day: '2-digit', month: '2-digit', year: 'numeric'
            });
            return `
                <div class="project-card" onclick="openProject(${p.id}, '${escHtml(p.name)}')" title="${escHtml(p.description || '')}">
                    <div class="project-card-header">
                        <span class="project-icon">📁</span>
                        <span class="project-name">${escHtml(p.name)}</span>
                    </div>
                    <div class="project-card-body">
                        ${p.description ? `<p class="project-desc">${escHtml(p.description)}</p>` : ''}
                        <div class="project-meta">
                            <span class="project-count">${p.content_count} items</span>
                            <span class="project-date">${date}</span>
                        </div>
                    </div>
                    <div class="project-card-actions" onclick="event.stopPropagation()">
                        <button class="btn btn-sm btn-danger-sm" onclick="deleteProject(${p.id}, '${escHtml(p.name)}')" title="Xoá project">🗑</button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = `<p class="negative">Lỗi tải projects: ${escHtml(err.message)}</p>`;
    }
}

async function createProject() {
    const nameEl = document.getElementById('newProjectName');
    const descEl = document.getElementById('newProjectDesc');
    const name = nameEl.value.trim();
    const desc = descEl.value.trim();

    if (!name) { toast('Vui lòng nhập tên project', 'error'); return; }

    try {
        const res = await fetch('/api/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description: desc }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        nameEl.value = '';
        descEl.value = '';
        toast(`Đã tạo project "${name}"`);
        loadProjects();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function deleteProject(id, name) {
    if (!confirm(`Xoá project "${name}" và tất cả content bên trong?`)) return;

    try {
        const res = await fetch(`/api/projects/${id}`, { method: 'DELETE' });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast('Đã xoá project');
        // If currently viewing this project, go back to list
        if (currentProjectId === id) {
            backToProjectList();
        }
        loadProjects();
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ── Open/Close Project Detail ──

async function openProject(id, name) {
    currentProjectId = id;
    currentProjectName = name;

    document.getElementById('projectListView').style.display = 'none';
    document.getElementById('projectDetailView').style.display = 'block';
    document.getElementById('projectDetailName').textContent = name;

    await loadContentItems(id);
}

function backToProjectList() {
    currentProjectId = null;
    currentProjectName = '';
    document.getElementById('projectDetailView').style.display = 'none';
    document.getElementById('projectListView').style.display = 'block';
    loadProjects();
}

// ── Content Items Table ──

async function loadContentItems(projectId) {
    const tbody = document.querySelector('#contentItemsTable tbody');
    const countEl = document.getElementById('projectItemCount');

    try {
        const res = await fetch(`/api/projects/${projectId}/items`);
        const json = await res.json();
        const items = json.items || [];

        countEl.textContent = `${items.length} items`;

        if (!items.length) {
            tbody.innerHTML = '<tr><td colspan="12" class="empty">Chưa có content nào. Tạo title & heading rồi bấm "Save to Project".</td></tr>';
            return;
        }

        // Store heading structures for viewing
        window._savedHeadings = {};
        window._savedItems = {};

        tbody.innerHTML = items.map((item, i) => {
            const statusMap = {
                'draft': 'status-draft', 'outline': 'status-outline',
                'review': 'status-review', 'approved': 'status-approved',
                'published': 'status-published', 'article_generated': 'status-generated'
            };
            const statusClass = statusMap[item.status] || 'status-draft';
            const statusLabel = item.status === 'article_generated' ? 'Article Generated' : item.status;
            const hasHeading = item.heading_structure && Object.keys(item.heading_structure).length > 0;
            const hasArticle = item.article_generated === 1 || (item.article_text && item.article_text.length > 0);
            const hasFeaturedImage = !!(item.featured_image_path);

            // Language display
            const langMap = {
                'vi': '🇻🇳', 'en': '🇬🇧', 'en-us': '🇺🇸', 'en-au': '🇦🇺',
                'fr': '🇫🇷', 'de': '🇩🇪', 'es': '🇪🇸', 'ja': '🇯🇵',
                'ko': '🇰🇷', 'zh': '🇨🇳', 'zh-tw': '🇹🇼', 'th': '🇹🇭',
                'pt': '🇧🇷', 'it': '🇮🇹'
            };
            const itemLang = item.language || 'vi';
            const langFlag = langMap[itemLang] || '🌐';
            const langLabel = itemLang.toUpperCase();

            // Featured image cell
            let featuredImageCell = '';
            if (hasFeaturedImage) {
                const imgName = item.featured_image_path.split(/[/\\]/).pop();
                const imgUrl = `/uploads/featured_images/${imgName}`;
                featuredImageCell = `
                    <div class="featured-img-cell">
                        <img src="${imgUrl}" alt="${escHtml(item.featured_image_alt || '')}" class="featured-thumb" onclick="openFeaturedImageModal(${item.id})">
                        <span class="featured-img-check">✅</span>
                    </div>`;
            } else {
                featuredImageCell = `<button class="btn btn-sm btn-upload-img" onclick="openFeaturedImageModal(${item.id})" title="Upload ảnh đại diện">🖼️ Upload</button>`;
            }

            // Publish status rendering
            const publishStatus = item.publish_status || 'draft';
            let publishCell = '';
            if (!hasArticle) {
                publishCell = '<span class="text-dim">—</span>';
            } else if (!hasFeaturedImage) {
                publishCell = '<span class="text-dim" title="Cần upload ảnh đại diện trước">⚠️ Cần ảnh</span>';
            } else if (publishStatus === 'published' && item.published_url) {
                publishCell = `<a href="${escHtml(item.published_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm btn-success-sm" title="${escHtml(item.published_url)}">🔗 Xem</a> <button class="btn btn-sm btn-outline" onclick="openPublishModal(${item.id})" title="Cập nhật / đăng lại (sửa ngôn ngữ, danh mục…)">🔄</button>`;
            } else if (publishStatus === 'published') {
                publishCell = `<span class="text-dim" title="Bài đã đăng nhưng chưa lưu được URL">✅ Đã đăng</span> <button class="btn btn-sm btn-outline" onclick="openPublishModal(${item.id})" title="Đăng lại để đồng bộ URL">🔄</button>`;
            } else if (publishStatus === 'publishing') {
                publishCell = '<span class="publish-loading"><span class="spinner-sm"></span> Đang đăng...</span>';
            } else if (publishStatus === 'failed') {
                publishCell = `<button class="btn btn-sm btn-danger-outline" onclick="openPublishModal(${item.id})" title="${escHtml(item.error_message || 'Thử lại')}">🔄 Retry</button>`;
            } else {
                publishCell = `<button class="btn btn-sm btn-publish" onclick="openPublishModal(${item.id})">🚀 Publish</button>`;
            }

            if (hasHeading) {
                window._savedHeadings[item.id] = { structure: item.heading_structure, title: item.title };
            }
            window._savedItems[item.id] = item;

            const hasHeadingForBatch = hasHeading && !hasArticle;
            return `
                <tr>
                    <td><input type="checkbox" class="item-checkbox" data-item-id="${item.id}" data-eligible="${hasHeadingForBatch ? 1 : 0}" onchange="updateSelectionCount()" ${hasHeadingForBatch ? '' : 'disabled title="Cần heading & chưa có bài"'}></td>
                    <td>${i + 1}</td>
                    <td title="${escHtml(item.keyword)}">${escHtml(truncate(item.keyword, 30))}</td>
                    <td title="${escHtml(item.title)}">${escHtml(truncate(item.title, 35))}</td>
                    <td>
                        <select class="lang-select" onchange="changeItemLanguage(${item.id}, this.value)">
                            ${[['vi','🇻🇳 Tiếng Việt'],['en','🇬🇧 English (UK)'],['en-us','🇺🇸 English (US)'],['en-au','🇦🇺 English (AU)'],['fr','🇫🇷 Français'],['de','🇩🇪 Deutsch'],['es','🇪🇸 Español'],['ja','🇯🇵 日本語'],['ko','🇰🇷 한국어'],['zh','🇨🇳 中文(简)'],['zh-tw','🇹🇼 中文(繁)'],['th','🇹🇭 ไทย'],['pt','🇧🇷 Português'],['it','🇮🇹 Italiano'],['ru','🇷🇺 Русский'],['id','🇮🇩 Indonesia']].map(([c,l])=>`<option value="${c}"${c===itemLang?' selected':''}>${l}</option>`).join('')}
                        </select>
                    </td>
                    <td>
                        ${hasHeading
                            ? `<div class="heading-actions-cell">
                                <button class="btn btn-sm" onclick="viewSavedHeadingById(${item.id})" title="Xem heading">📐 Xem</button>
                                <button class="btn btn-sm btn-edit" onclick="editSavedHeadingById(${item.id})" title="Sửa heading">✏️</button>
                               </div>`
                            : '<span class="text-dim">—</span>'
                        }
                    </td>
                    <td>
                        ${hasArticle
                            ? `<button class="btn btn-sm btn-success-sm" onclick="viewArticle(${item.id})" title="Xem bài viết">📝 Xem</button>`
                            : (hasHeading
                                ? `<button class="btn btn-sm btn-accent-sm" onclick="generateArticleForItem(${item.id}, this)" title="Tạo bài blog">🚀 Viết bài</button>`
                                : '<span class="text-dim">—</span>'
                              )
                        }
                    </td>
                    <td>${featuredImageCell}</td>
                    <td>${publishCell}</td>
                    <td>${(() => {
                        const slug = item.url_slug || (item.published_url ? item.published_url.replace(/\/$/, '').split('/').pop() : '');
                        if (!slug) return `<span class="text-dim" ondblclick="editSlug(${item.id}, this)" title="Double-click để thêm slug" style="cursor:pointer;">+</span>`;
                        return `<span class="url-slug-cell" title="Double-click để sửa" ondblclick="editSlug(${item.id}, this)" style="cursor:pointer;font-size:0.78rem;color:#64b5f6;max-width:120px;display:inline-block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(truncate(slug,22))}</span>`;
                    })()}</td>
                    <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
                    <td>
                        <button class="btn btn-sm btn-danger-sm" onclick="deleteContentItem(${item.id})" title="Xoá">🗑</button>
                    </td>
                </tr>
            `;
        }).join('');
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="12" class="negative">Lỗi: ${escHtml(err.message)}</td></tr>`;
    }
}

function viewSavedHeadingById(itemId) {
    const data = window._savedHeadings?.[itemId];
    if (!data) { toast('Không tìm thấy heading data', 'error'); return; }
    viewSavedHeading(data.structure, data.title);
}

function viewSavedHeading(structure, title) {
    document.getElementById('headingModalTitle').textContent = `📐 Heading: ${truncate(title, 50)}`;
    // Re-render heading structure in modal
    lastHeadingResult = structure;
    renderHeadingStructure(structure, title, '');
    document.getElementById('headingModal').style.display = 'flex';
}

function editSlug(itemId, el) {
    const item = window._savedItems?.[itemId];
    if (!item) return;
    const currentSlug = item.url_slug || (item.published_url ? item.published_url.replace(/\/$/, '').split('/').pop() : '');
    const td = el.closest('td');

    td.innerHTML = `
        <input id="slugInput_${itemId}" type="text" value="${escHtml(currentSlug)}"
            style="width:110px;font-size:0.78rem;padding:2px 5px;background:#1e293b;color:#e2e8f0;border:1px solid #64b5f6;border-radius:4px;outline:none;"
            onkeydown="handleSlugKey(event,${itemId})"
            onblur="cancelSlugEdit(${itemId})">
        <button class="btn btn-sm" style="padding:1px 5px;font-size:0.72rem;" onmousedown="event.preventDefault();saveSlug(${itemId})">✓</button>
    `;
    document.getElementById(`slugInput_${itemId}`)?.focus();
}

async function saveSlug(itemId) {
    const input = document.getElementById(`slugInput_${itemId}`);
    if (!input) return;
    const newSlug = input.value.trim();
    try {
        const res = await fetch(`/api/content-items/${itemId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url_slug: newSlug }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Lỗi');
        if (window._savedItems?.[itemId]) window._savedItems[itemId].url_slug = newSlug;
        toast('Đã lưu slug');
        if (currentProjectId) loadContentItems(currentProjectId);
    } catch (e) {
        toast(e.message, 'error');
        if (currentProjectId) loadContentItems(currentProjectId);
    }
}

function handleSlugKey(e, itemId) {
    if (e.key === 'Enter') { e.preventDefault(); saveSlug(itemId); }
    if (e.key === 'Escape') { if (currentProjectId) loadContentItems(currentProjectId); }
}

function cancelSlugEdit(itemId) {
    // small delay so saveSlug mousedown fires first
    setTimeout(() => {
        if (document.getElementById(`slugInput_${itemId}`)) {
            if (currentProjectId) loadContentItems(currentProjectId);
        }
    }, 150);
}

async function changeItemLanguage(itemId, newLang) {
    const item = window._savedItems?.[itemId];
    if (!item) return;
    const currentLang = item.language || 'vi';
    if (newLang === currentLang) return;

    try {
        const res = await fetch(`/api/content-items/${itemId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: newLang }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast(`Đã đổi ngôn ngữ → ${newLang}`);
        if (window._savedItems[itemId]) window._savedItems[itemId].language = newLang;
    } catch (err) {
        toast(err.message, 'error');
        // Revert select
        if (currentProjectId) loadContentItems(currentProjectId);
    }
}

async function deleteContentItem(itemId) {
    if (!confirm('Xoá content item này?')) return;

    try {
        const res = await fetch(`/api/content-items/${itemId}`, { method: 'DELETE' });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast('Đã xoá content item');
        if (currentProjectId) loadContentItems(currentProjectId);
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ── Manual Blog Entry Modal ──

function openManualEntryModal() {
    if (!currentProjectId) {
        toast('Chưa chọn project.', 'error');
        return;
    }
    // Reset form
    document.getElementById('manualKeyword').value = '';
    document.getElementById('manualTitle').value = '';
    document.getElementById('manualArticleContent').value = '';
    document.getElementById('manualStatus').value = 'article_generated';
    document.getElementById('manualPriority').value = '0';
    // Set language from project default
    const defaultLang = document.getElementById('projectDefaultLang')?.value || 'vi';
    document.getElementById('manualLang').value = defaultLang;
    document.getElementById('manualEntryModal').style.display = 'flex';
}

function closeManualEntryModal() {
    document.getElementById('manualEntryModal').style.display = 'none';
    // Reset heading context so it doesn't leak to next manual entry
    manualEntryHeadingContext = null;
    document.getElementById('manualHeadingPreview').style.display = 'none';
    _resetHeadingModalButtons();
}

async function submitManualEntry() {
    const keyword = document.getElementById('manualKeyword').value.trim();
    const title = document.getElementById('manualTitle').value.trim();
    const language = document.getElementById('manualLang').value;
    const articleContent = document.getElementById('manualArticleContent').value.trim();
    const status = document.getElementById('manualStatus').value;
    const priority = parseInt(document.getElementById('manualPriority').value) || 0;

    if (!keyword) { toast('Vui lòng nhập keyword.', 'error'); return; }
    if (!title) { toast('Vui lòng nhập title.', 'error'); return; }

    // Attach heading structure if it was generated for the same keyword+title
    const hasMatchingHeading = lastHeadingResult && manualEntryHeadingContext &&
        manualEntryHeadingContext.keyword === keyword &&
        manualEntryHeadingContext.title === title;

    const btn = document.getElementById('btnSubmitManual');
    const origText = btn.textContent;
    btn.disabled = true;
    btn.textContent = '⏳ Đang lưu...';

    try {
        const payload = { keyword, title, language, status, priority };
        if (hasMatchingHeading) payload.heading_structure = lastHeadingResult;

        // Detect if content is HTML or plain text
        if (articleContent) {
            const isHtml = /<[a-z][\s\S]*>/i.test(articleContent);
            if (isHtml) {
                payload.article_html = articleContent;
                // Create a plain text version by stripping tags
                const tmp = document.createElement('div');
                tmp.innerHTML = articleContent;
                payload.article_text = tmp.textContent || tmp.innerText || '';
            } else {
                payload.article_text = articleContent;
                // Convert plain text to basic HTML
                payload.article_html = articleContent.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('\n');
            }
        }

        const res = await fetch(`/api/projects/${currentProjectId}/items`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Lỗi không xác định');

        const savedHeading = hasMatchingHeading ? ' (kèm heading structure)' : '';
        toast(`✅ Đã thêm bài viết thành công!${savedHeading}`);
        closeManualEntryModal();
        await loadContentItems(currentProjectId);
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = origText;
    }
}

// ── Save to Project Modal ──

function openSaveToProjectModal() {
    if (!lastHeadingResult || !lastTitleGenData) {
        toast('Chưa có heading structure để lưu.', 'error');
        return;
    }

    // Cache the current data
    saveKeywordCache = lastTitleGenData.keyword || '';
    // Find the title that was used - it's in the modal title
    const modalTitle = document.getElementById('headingModalTitle').textContent;
    saveTitleCache = lastHeadingResult.h1 || modalTitle.replace('📐 Heading Structure: ', '').replace('📐 Heading: ', '');
    saveHeadingCache = lastHeadingResult;

    document.getElementById('saveKeyword').value = saveKeywordCache;
    document.getElementById('saveTitle').value = saveTitleCache;

    // Load projects into dropdown
    loadProjectDropdown();

    document.getElementById('saveToProjectModal').style.display = 'flex';
}

function closeSaveToProjectModal() {
    document.getElementById('saveToProjectModal').style.display = 'none';
}

async function loadProjectDropdown() {
    const select = document.getElementById('saveProjectSelect');
    try {
        const res = await fetch('/api/projects');
        const json = await res.json();
        const projects = json.projects || [];

        let options = '<option value="">-- Chọn project --</option>';
        for (const p of projects) {
            options += `<option value="${p.id}">${escHtml(p.name)} (${p.content_count} items)</option>`;
        }
        select.innerHTML = options;
    } catch (err) {
        select.innerHTML = '<option value="">Lỗi tải projects</option>';
    }
}

async function createProjectFromModal() {
    const nameEl = document.getElementById('saveNewProjectName');
    const name = nameEl.value.trim();
    if (!name) { toast('Nhập tên project', 'error'); return; }

    try {
        const res = await fetch('/api/projects', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        nameEl.value = '';
        toast(`Đã tạo project "${name}"`);

        // Refresh dropdown and select the new project
        await loadProjectDropdown();
        document.getElementById('saveProjectSelect').value = json.project.id;
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function saveContentToProject() {
    const projectId = document.getElementById('saveProjectSelect').value;
    if (!projectId) {
        toast('Vui lòng chọn project', 'error');
        return;
    }

    const status = document.getElementById('saveStatus').value;
    const priority = parseInt(document.getElementById('savePriority').value);

    const btn = document.getElementById('btnConfirmSave');
    btn.disabled = true;

    // Read the CURRENT values from the input fields (user may have edited them)
    const currentTitle = document.getElementById('saveTitle').value.trim() || saveTitleCache;
    const currentKeyword = document.getElementById('saveKeyword').value.trim() || saveKeywordCache;

    try {
        // Get language from Title Rewrites page selector or project default
        const language = document.getElementById('rewriteLang')?.value
                      || document.getElementById('projectDefaultLang')?.value
                      || 'vi';
        const res = await fetch(`/api/projects/${projectId}/items`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                keyword: currentKeyword,
                title: currentTitle,
                heading_structure: saveHeadingCache,
                status,
                priority,
                language,
            }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        toast(`Đã lưu vào project thành công! ✓`);
        closeSaveToProjectModal();

        // Also refresh project list if visible
        loadProjects();
    } catch (err) {
        toast(err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

// Close save modal on backdrop click
document.addEventListener('click', (e) => {
    if (e.target.id === 'saveToProjectModal') closeSaveToProjectModal();
    if (e.target.id === 'headingEditModal') closeHeadingEditModal();
    if (e.target.id === 'articleModal') closeArticleModal();
});

// ══════════════════════════════════════════════
// Heading Editor — Edit saved heading structures
// ══════════════════════════════════════════════

let editingItemId = null;
let editingHeadingData = null;

function editSavedHeadingById(itemId) {
    const data = window._savedHeadings?.[itemId];
    if (!data) { toast('Không tìm thấy heading data', 'error'); return; }
    editingItemId = itemId;
    editingHeadingData = JSON.parse(JSON.stringify(data.structure)); // deep copy
    openHeadingEditModal(data.title);
}

function openHeadingEditModal(title) {
    document.getElementById('headingEditModalTitle').textContent = `✏️ Edit: ${truncate(title, 50)}`;
    document.getElementById('headingEditTitleInput').value = editingHeadingData.h1 || title || '';
    renderEditableHeadings();
    document.getElementById('headingEditModal').style.display = 'flex';
}

function closeHeadingEditModal() {
    document.getElementById('headingEditModal').style.display = 'none';
    editingItemId = null;
    editingHeadingData = null;
}

function renderEditableHeadings() {
    const body = document.getElementById('headingEditBody');
    if (!editingHeadingData) { body.innerHTML = '<p class="empty">Không có dữ liệu</p>'; return; }

    const sections = editingHeadingData.sections || [];
    const faq = editingHeadingData.faq || [];

    let html = '';

    // Sections
    sections.forEach((sec, i) => {
        const nestClass = sec.level === 'h3' ? 'edit-nested' : (sec.level === 'h4' ? 'edit-nested edit-deep' : '');
        html += `
            <div class="edit-node edit-${sec.level} ${nestClass}" data-index="${i}">
                <span class="edit-tag tag-${sec.level}">${sec.level.toUpperCase()}</span>
                <input type="text" class="edit-heading-input" value="${escHtml(sec.heading || '')}"
                       onchange="editingHeadingData.sections[${i}].heading = this.value">
                <select class="edit-level-select" onchange="changeHeadingLevel(${i}, this.value)">
                    <option value="h2" ${sec.level === 'h2' ? 'selected' : ''}>H2</option>
                    <option value="h3" ${sec.level === 'h3' ? 'selected' : ''}>H3</option>
                    <option value="h4" ${sec.level === 'h4' ? 'selected' : ''}>H4</option>
                </select>
                <button class="btn btn-sm btn-icon-only" onclick="moveHeadingUp(${i})" title="Di chuyển lên" ${i === 0 ? 'disabled' : ''}>↑</button>
                <button class="btn btn-sm btn-icon-only" onclick="moveHeadingDown(${i})" title="Di chuyển xuống" ${i === sections.length - 1 ? 'disabled' : ''}>↓</button>
                <button class="btn btn-sm btn-danger-sm" onclick="removeHeadingNode(${i})" title="Xoá">✕</button>
            </div>
        `;
    });

    // FAQ
    if (faq.length) {
        html += '<div class="edit-section-divider">❓ FAQ</div>';
        faq.forEach((f, i) => {
            html += `
                <div class="edit-node edit-faq" data-faq-index="${i}">
                    <span class="edit-tag tag-faq">FAQ</span>
                    <input type="text" class="edit-heading-input" value="${escHtml(f.question || '')}"
                           onchange="editingHeadingData.faq[${i}].question = this.value"
                           placeholder="Câu hỏi FAQ...">
                    <button class="btn btn-sm btn-danger-sm" onclick="removeFaqNode(${i})" title="Xoá FAQ">✕</button>
                </div>
            `;
        });
    }

    body.innerHTML = html;
    document.getElementById('headingEditInfo').textContent = `${sections.length} sections · ${faq.length} FAQ`;
}

function changeHeadingLevel(index, newLevel) {
    editingHeadingData.sections[index].level = newLevel;
    renderEditableHeadings();
}

function moveHeadingUp(index) {
    if (index <= 0) return;
    const sec = editingHeadingData.sections;
    [sec[index - 1], sec[index]] = [sec[index], sec[index - 1]];
    renderEditableHeadings();
}

function moveHeadingDown(index) {
    const sec = editingHeadingData.sections;
    if (index >= sec.length - 1) return;
    [sec[index], sec[index + 1]] = [sec[index + 1], sec[index]];
    renderEditableHeadings();
}

function addHeadingNode(level) {
    if (!editingHeadingData) return;
    if (!editingHeadingData.sections) editingHeadingData.sections = [];
    editingHeadingData.sections.push({
        level: level,
        heading: '',
        eeat_signal: 'expertise',
        purpose: '',
        content_type: 'text',
        key_points: [],
        word_count_target: level === 'h2' ? 300 : 150,
        brand_knowledge_used: false,
        is_product_section: false,
    });
    renderEditableHeadings();
    // Focus the new input
    setTimeout(() => {
        const inputs = document.querySelectorAll('.edit-heading-input');
        if (inputs.length) inputs[inputs.length - 1].focus();
    }, 50);
}

function addFaqNode() {
    if (!editingHeadingData) return;
    if (!editingHeadingData.faq) editingHeadingData.faq = [];
    editingHeadingData.faq.push({
        question: '',
        answer_outline: '',
        source: 'generated',
        schema_ready: true,
    });
    renderEditableHeadings();
}

function removeHeadingNode(index) {
    editingHeadingData.sections.splice(index, 1);
    renderEditableHeadings();
}

function removeFaqNode(index) {
    editingHeadingData.faq.splice(index, 1);
    renderEditableHeadings();
}

async function saveEditedHeading() {
    if (!editingItemId || !editingHeadingData) {
        toast('Không có dữ liệu để lưu', 'error');
        return;
    }

    // Validate: at least some heading content
    const validSections = editingHeadingData.sections.filter(s => s.heading && s.heading.trim());
    if (validSections.length === 0) {
        toast('Cần ít nhất 1 heading có nội dung', 'error');
        return;
    }

    // Clean up empty headings
    editingHeadingData.sections = validSections;
    editingHeadingData.faq = (editingHeadingData.faq || []).filter(f => f.question && f.question.trim());

    // Update meta counts
    editingHeadingData.meta = editingHeadingData.meta || {};
    editingHeadingData.meta.total_sections = editingHeadingData.sections.length;
    editingHeadingData.meta.h2_count = editingHeadingData.sections.filter(s => s.level === 'h2').length;
    editingHeadingData.meta.h3_count = editingHeadingData.sections.filter(s => s.level === 'h3').length;
    editingHeadingData.meta.h4_count = editingHeadingData.sections.filter(s => s.level === 'h4').length;

    // Get the updated title from dedicated title input
    const updatedTitle = (document.getElementById('headingEditTitleInput')?.value || '').trim();
    if (updatedTitle) editingHeadingData.h1 = updatedTitle;

    try {
        const payload = {
            heading_structure: editingHeadingData,
            status: 'approved',
            article_text: '',
            article_html: '',
            internal_links: [],
            article_generated: 0,
        };
        // Also update title in DB if H1 was changed
        if (updatedTitle) {
            payload.title = updatedTitle;
        }

        const res = await fetch(`/api/content-items/${editingItemId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        // Update local cache (both structure AND title)
        if (window._savedHeadings && window._savedHeadings[editingItemId]) {
            window._savedHeadings[editingItemId].structure = editingHeadingData;
            if (updatedTitle) {
                window._savedHeadings[editingItemId].title = updatedTitle;
            }
        }

        toast('Heading đã được lưu & approve! Bài viết cũ đã được reset — bấm Viết bài để tạo lại. ✅');
        closeHeadingEditModal();

        // Refresh content items table
        if (currentProjectId) loadContentItems(currentProjectId);
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ══════════════════════════════════════════════
// Article Generation & Viewing
// ══════════════════════════════════════════════

let currentArticleData = null;
let currentArticleItemId = null;

async function generateArticleForItem(itemId, btnEl) {
    const item = window._savedItems?.[itemId];
    if (!item) { toast('Không tìm thấy item', 'error'); return; }

    const itemLang = item.language || 'vi';
    const langNames = {'vi':'Tiếng Việt','en':'English','en-us':'English (US)','en-au':'English (AU)','fr':'Français','de':'Deutsch','es':'Español','ja':'日本語','ko':'한국어','zh':'中文','zh-tw':'中文(繁)','th':'ไทย','pt':'Português','it':'Italiano'};
    const langName = langNames[itemLang] || itemLang;

    if (!confirm(`Tạo bài blog cho "${truncate(item.title, 50)}"?\nNgôn ngữ: ${langName}\nQuá trình này có thể mất 30-60 giây.`)) return;

    // Show loading in the button cell
    const btn = btnEl || document.querySelector(`button[onclick*="generateArticleForItem(${itemId}"]`);
    const originalHtml = btn ? btn.innerHTML : '';
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-sm"></span> Đang viết...';
    }

    try {
        const language = itemLang;
        const res = await fetch(`/api/content-items/${itemId}/generate-article`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        toast(`Bài viết đã được tạo! ${json.word_count} từ, ${json.internal_links?.length || 0} links ✅`);

        // Refresh table
        if (currentProjectId) await loadContentItems(currentProjectId);

        // Auto-open article view
        currentArticleData = json;
        currentArticleItemId = itemId;
        openArticleModal(item.title);
    } catch (err) {
        toast(err.message, 'error');
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = originalHtml;
        }
    }
}

// ── Batch Article Generation ──

let currentBatchJobId = null;
let batchPollInterval = null;

async function startBatchGenerate() {
    if (!currentProjectId) { toast('Chưa chọn project.', 'error'); return; }

    // Count eligible items
    const items = Object.values(window._savedItems || {});
    const eligible = items.filter(it => {
        const hasHeading = it.heading_structure && it.heading_structure.sections && it.heading_structure.sections.length > 0;
        const hasArticle = it.article_generated === 1 && it.article_text;
        return hasHeading && !hasArticle;
    });

    if (!eligible.length) {
        toast('Không có bài nào cần tạo. Tất cả đã có bài viết hoặc chưa có heading.', 'error');
        return;
    }

    const names = eligible.slice(0, 5).map(it => `• ${truncate(it.title, 40)}`).join('\n');
    const more = eligible.length > 5 ? `\n... và ${eligible.length - 5} bài khác` : '';

    if (!confirm(`Viết bài hàng loạt cho ${eligible.length} bài?\n\n${names}${more}\n\nMỗi bài mất ~30-60 giây. Tổng ước tính: ${eligible.length * 0.5}-${eligible.length} phút.`)) return;

    try {
        const res = await fetch(`/api/projects/${currentProjectId}/batch-generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skip_existing: true }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Lỗi');

        currentBatchJobId = json.job_id;
        toast(`🚀 Bắt đầu viết ${json.total} bài trong nền!`);

        // Show progress container
        showBatchProgress(json.total);

        // Start polling
        batchPollInterval = setInterval(() => pollBatchStatus(), 3000);

    } catch (err) {
        toast(err.message, 'error');
    }
}

function showBatchProgress(total) {
    const container = document.getElementById('batchProgressContainer');
    container.style.display = 'block';
    document.getElementById('batchProgressStats').textContent = `0/${total}`;
    document.getElementById('batchProgressBar').style.width = '0%';
    document.getElementById('batchProgressBar').textContent = '0%';
    document.getElementById('batchCurrentItem').textContent = 'Đang chuẩn bị...';
    document.getElementById('btnCancelBatch').style.display = 'inline-block';
    document.getElementById('batchProgressTitle').textContent = '🚀 Viết bài hàng loạt...';
}

async function pollBatchStatus() {
    if (!currentBatchJobId) return;

    try {
        const res = await fetch(`/api/batch-jobs/${currentBatchJobId}`);
        const job = await res.json();
        if (!res.ok) {
            clearInterval(batchPollInterval);
            return;
        }

        const done = job.completed + job.failed;
        const pct = job.total > 0 ? Math.round((done / job.total) * 100) : 0;

        document.getElementById('batchProgressStats').textContent = `${done}/${job.total} (${job.failed > 0 ? job.failed + ' lỗi' : 'OK'})`;
        document.getElementById('batchProgressBar').style.width = pct + '%';
        document.getElementById('batchProgressBar').textContent = pct + '%';

        if (job.current_item) {
            document.getElementById('batchCurrentItem').textContent = `📝 Đang viết: ${job.current_item}`;
        }

        // Job finished?
        if (job.status === 'completed' || job.status === 'cancelled') {
            clearInterval(batchPollInterval);
            batchPollInterval = null;

            const title = job.status === 'cancelled' ? '⏹ Đã huỷ' : '✅ Hoàn thành';
            document.getElementById('batchProgressTitle').textContent = `${title}: ${job.completed} bài thành công, ${job.failed} lỗi`;
            document.getElementById('batchCurrentItem').textContent = '';
            document.getElementById('btnCancelBatch').style.display = 'none';

            // Change progress bar color on complete
            const bar = document.getElementById('batchProgressBar');
            bar.style.width = '100%';
            bar.textContent = title;
            if (job.status === 'cancelled') {
                bar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            } else if (job.failed > 0) {
                bar.style.background = 'linear-gradient(90deg, #f59e0b, #34d399)';
            }

            toast(`${title}: ${job.completed}/${job.total} bài viết`);

            // Refresh content table
            if (currentProjectId) loadContentItems(currentProjectId);

            // Auto-hide after 10s
            setTimeout(() => {
                document.getElementById('batchProgressContainer').style.display = 'none';
            }, 10000);

            currentBatchJobId = null;
        }
    } catch (err) {
        // Network error — keep polling
    }
}

async function cancelBatchJob() {
    if (!currentBatchJobId) return;
    if (!confirm('Huỷ viết bài hàng loạt? Bài đang viết sẽ được hoàn thành trước khi dừng.')) return;

    try {
        await fetch(`/api/batch-jobs/${currentBatchJobId}/cancel`, { method: 'POST' });
        toast('Đã gửi lệnh huỷ. Đợi bài hiện tại hoàn thành...');
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ── Selection-based Batch Generation ──

function toggleSelectAll(masterCheckbox) {
    const checkboxes = document.querySelectorAll('.item-checkbox');
    checkboxes.forEach(cb => {
        if (!cb.disabled) cb.checked = masterCheckbox.checked;
    });
    updateSelectionCount();
}

function updateSelectionCount() {
    const checked = document.querySelectorAll('.item-checkbox:checked');
    const count = checked.length;
    const btn = document.getElementById('btnBatchSelected');
    const countEl = document.getElementById('selectedCount');
    if (count > 0) {
        btn.style.display = 'inline-flex';
        countEl.textContent = count;
    } else {
        btn.style.display = 'none';
    }
    // Sync master checkbox
    const all = document.querySelectorAll('.item-checkbox:not(:disabled)');
    const master = document.getElementById('selectAllItems');
    if (master) {
        master.checked = all.length > 0 && checked.length === all.length;
        master.indeterminate = checked.length > 0 && checked.length < all.length;
    }
}

function getSelectedItemIds() {
    const checked = document.querySelectorAll('.item-checkbox:checked');
    return Array.from(checked).map(cb => parseInt(cb.dataset.itemId));
}

async function startBatchGenerateSelected() {
    if (!currentProjectId) { toast('Chưa chọn project.', 'error'); return; }

    const selectedIds = getSelectedItemIds();
    if (!selectedIds.length) {
        toast('Chưa chọn bài nào!', 'error');
        return;
    }

    // Filter eligible items (must have heading, no article yet)
    const eligible = selectedIds.filter(id => {
        const item = window._savedItems?.[id];
        if (!item) return false;
        const hasHeading = item.heading_structure && item.heading_structure.sections && item.heading_structure.sections.length > 0;
        const hasArticle = item.article_generated === 1 && item.article_text;
        return hasHeading && !hasArticle;
    });

    if (!eligible.length) {
        toast('Các bài đã chọn đều đã có bài viết hoặc chưa có heading.', 'error');
        return;
    }

    const names = eligible.slice(0, 5).map(id => {
        const item = window._savedItems[id];
        return `• ${truncate(item.title, 40)}`;
    }).join('\n');
    const more = eligible.length > 5 ? `\n... và ${eligible.length - 5} bài khác` : '';

    if (!confirm(`Viết bài cho ${eligible.length} bài đã chọn?\n\n${names}${more}\n\nMỗi bài mất ~30-60 giây. Tổng ước tính: ${eligible.length * 0.5}-${eligible.length} phút.`)) return;

    try {
        const res = await fetch(`/api/projects/${currentProjectId}/batch-generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_ids: eligible, skip_existing: true }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || 'Lỗi');

        currentBatchJobId = json.job_id;
        toast(`🚀 Bắt đầu viết ${json.total} bài đã chọn!`);

        showBatchProgress(json.total);
        batchPollInterval = setInterval(() => pollBatchStatus(), 3000);

        // Uncheck all
        document.querySelectorAll('.item-checkbox:checked').forEach(cb => cb.checked = false);
        updateSelectionCount();

    } catch (err) {
        toast(err.message, 'error');
    }
}

async function viewArticle(itemId) {
    const item = window._savedItems?.[itemId];
    if (!item) { toast('Không tìm thấy item', 'error'); return; }

    try {
        const res = await fetch(`/api/content-items/${itemId}/article`);
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        if (!json.has_article) {
            toast('Bài viết chưa được tạo', 'error');
            return;
        }

        currentArticleData = json;
        currentArticleItemId = itemId;
        openArticleModal(item.title);
    } catch (err) {
        toast(err.message, 'error');
    }
}

function openArticleModal(title) {
    document.getElementById('articleModalTitle').textContent = `📝 ${truncate(title, 50)}`;

    // Meta bar
    const meta = document.getElementById('articleMeta');
    const data = currentArticleData;

    meta.innerHTML = `
        <div class="article-meta-item"><strong>${data.word_count || 0}</strong> từ</div>
        <div class="article-meta-item"><strong>${data.internal_links?.length || 0}</strong> internal links</div>
        <div class="article-meta-item"><strong>${Math.ceil((data.word_count || 0) / 200)}</strong> phút đọc</div>
    `;

    // Read view (render article HTML)
    const readView = document.getElementById('articleReadView');
    readView.innerHTML = data.article_html || renderArticleReadMode(data.article_text || '');

    // HTML view (for edit mode)
    const htmlCode = document.getElementById('articleEditCode');
    htmlCode.value = data.article_html || '';

    // Reset to read mode
    switchArticleMode('read');

    document.getElementById('articleModal').style.display = 'flex';
}

function closeArticleModal() {
    document.getElementById('articleModal').style.display = 'none';
    currentArticleData = null;
    currentArticleItemId = null;
}

function switchArticleMode(mode) {
    const readBtn = document.getElementById('btnReadMode');
    const editBtn = document.getElementById('btnEditMode');
    const readView = document.getElementById('articleReadView');
    const editView = document.getElementById('articleEditView');
    const saveBtn = document.getElementById('btnSaveArticle');

    if (readBtn) readBtn.classList.remove('active');
    if (editBtn) editBtn.classList.remove('active');
    if (readView) readView.style.display = 'none';
    if (editView) editView.style.display = 'none';
    if (saveBtn) saveBtn.style.display = 'none';

    if (mode === 'read') {
        if (readBtn) readBtn.classList.add('active');
        if (readView) readView.style.display = 'block';
    } else if (mode === 'edit') {
        if (editBtn) editBtn.classList.add('active');
        if (editView) editView.style.display = 'block';
        if (saveBtn) saveBtn.style.display = 'inline-flex';
        const editCode = document.getElementById('articleEditCode');
        if (editCode) editCode.value = currentArticleData?.article_html || '';
    }
}

function markdownToHtml(md) {
    if (!md) return '';
    let html = md;

    // Headings
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');

    // Bold + italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2">$1</a>');

    // Blockquotes
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

    // Unordered lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Paragraphs: wrap non-tag lines
    const lines = html.split('\n');
    const result = [];
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) { result.push(''); continue; }
        if (/^<(h[1-6]|ul|ol|li|blockquote|table|thead|tbody|tr|th|td|div|p)/.test(trimmed)) {
            result.push(trimmed);
        } else {
            result.push(`<p>${trimmed}</p>`);
        }
    }
    return result.join('\n');
}

async function saveArticleEdit() {
    if (!currentArticleItemId) { toast('Không xác định được bài viết', 'error'); return; }

    const editCode = document.getElementById('articleEditCode');
    const newHtml = editCode.value;

    // Extract plain text from HTML for article_text
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = newHtml;
    const newText = tempDiv.textContent || tempDiv.innerText || '';

    // Disable all save buttons
    const saveBtn = document.getElementById('btnSaveArticle');
    const saveBtnInline = document.getElementById('btnSaveArticleInline');
    const statusEl = document.getElementById('editSaveStatus');
    const origFooter = saveBtn ? saveBtn.innerHTML : '';
    const origInline = saveBtnInline ? saveBtnInline.innerHTML : '';

    if (saveBtn) { saveBtn.disabled = true; saveBtn.innerHTML = '<span class="spinner-sm"></span> Đang lưu...'; }
    if (saveBtnInline) { saveBtnInline.disabled = true; saveBtnInline.innerHTML = '<span class="spinner-sm"></span> Đang lưu...'; }
    if (statusEl) statusEl.textContent = '';

    try {
        const res = await fetch(`/api/content-items/${currentArticleItemId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                article_text: newText,
                article_html: newHtml,
            }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        // Update local data
        currentArticleData.article_text = newText;
        currentArticleData.article_html = newHtml;
        currentArticleData.word_count = newText.split(/\s+/).filter(w => w).length;

        // Update read view from HTML
        document.getElementById('articleReadView').innerHTML = newHtml;

        // Update meta bar
        const meta = document.getElementById('articleMeta');
        meta.innerHTML = `
            <div class="article-meta-item"><strong>${currentArticleData.word_count}</strong> từ</div>
            <div class="article-meta-item"><strong>${currentArticleData.internal_links?.length || 0}</strong> internal links</div>
            <div class="article-meta-item"><strong>${Math.ceil(currentArticleData.word_count / 200)}</strong> phút đọc</div>
        `;

        toast('Đã lưu thay đổi bài viết! ✅');
        if (statusEl) { statusEl.textContent = '✅ Đã lưu'; statusEl.style.color = '#48bb78'; }

        // Refresh table
        if (currentProjectId) loadContentItems(currentProjectId);
    } catch (err) {
        toast('Lỗi lưu: ' + err.message, 'error');
        if (statusEl) { statusEl.textContent = '❌ ' + err.message; statusEl.style.color = '#fc8181'; }
    } finally {
        if (saveBtn) { saveBtn.disabled = false; saveBtn.innerHTML = origFooter; }
        if (saveBtnInline) { saveBtnInline.disabled = false; saveBtnInline.innerHTML = origInline; }
    }
}

function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function renderArticleReadMode(markdownText) {
    if (!markdownText) return '<p class="empty">Chưa có nội dung bài viết</p>';

    let html = markdownText;

    // Convert headings
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" style="color: #00d4aa;">$1</a>');

    // Blockquotes
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

    // Convert markdown tables to styled HTML
    html = html.replace(/(\|.+\|[\r\n]+\|[-| :]+\|[\r\n]+((\|.+\|[\r\n]*)+))/gm, function(tableBlock) {
        const rows = tableBlock.trim().split('\n').filter(r => r.trim());
        if (rows.length < 2) return tableBlock;
        const headers = rows[0].split('|').filter(c => c.trim()).map(c => c.trim());
        let tbl = '<table style="width:100%;border-collapse:collapse;margin:20px 0;border:1px solid #444;"><thead><tr style="background:#2a2d35;">';
        headers.forEach(h => { tbl += `<th style="padding:10px 14px;border-bottom:2px solid #555;color:#e0e0e0;font-weight:bold;text-align:center;">${h}</th>`; });
        tbl += '</tr></thead><tbody>';
        for (let i = 2; i < rows.length; i++) {
            const cells = rows[i].split('|').filter(c => c.trim()).map(c => c.trim());
            const bg = i % 2 === 0 ? '#1e2028' : '#22252e';
            tbl += `<tr style="background:${bg};border-bottom:1px solid #333;">`;
            cells.forEach((c, ci) => {
                const fw = ci === 0 ? 'font-weight:bold;' : '';
                tbl += `<td style="padding:10px 14px;color:#ccc;${fw}">${c}</td>`;
            });
            tbl += '</tr>';
        }
        tbl += '</tbody></table>';
        return tbl;
    });

    // Convert lines to paragraphs and nested lists
    const lines = html.split('\n');
    const result = [];
    let inOl = false;
    let olLiOpen = false;       // Unclosed <li> inside <ol>
    let inNestedUl = false;     // <ul> nested within <ol><li>
    let inStandaloneUl = false; // Top-level <ul>

    function formatOlLiContent(content) {
        // Split '<strong>Title:</strong> body' onto separate lines
        const m = content.match(/^(<strong>[^<]+<\/strong>)\s+(.*)/s);
        if (m && m[2].trim()) return `${m[1]}\n    ${m[2].trim()}`;
        return content;
    }

    for (const line of lines) {
        const trimmed = line.trim();
        const isUlItem = /^[-*+] /.test(trimmed);
        const isOlItem = /^\d+\.\s/.test(trimmed);
        const isEmpty = !trimmed;
        const isHeading = trimmed.startsWith('<h');

        // ── Closing logic (only on non-empty lines) ──
        if (!isEmpty) {
            if (inNestedUl && !isUlItem) {
                result.push('    </ul>');
                inNestedUl = false;
            }
            if (olLiOpen && !isUlItem) {
                result.push('  </li>');
                olLiOpen = false;
            }
            if (inOl && !isOlItem && !isUlItem) {
                result.push('</ol>');
                inOl = false;
            }
            if (inStandaloneUl && !isUlItem) {
                result.push('</ul>');
                inStandaloneUl = false;
            }
        }

        // ── Ordered list item ──
        if (isOlItem) {
            if (!inOl) {
                result.push('<ol>');
                inOl = true;
            }
            const content = trimmed.replace(/^\d+\.\s/, '');
            const formatted = formatOlLiContent(content);
            result.push(`\n  <li>\n    ${formatted}`);
            olLiOpen = true;
            continue;
        }

        // ── Unordered list item ──
        if (isUlItem) {
            const content = trimmed.replace(/^[-*+] /, '');
            if (inOl && olLiOpen) {
                // Nest inside current <ol><li>
                if (!inNestedUl) {
                    result.push('    <ul>');
                    inNestedUl = true;
                }
                result.push(`      <li>${content}</li>`);
            } else {
                // Standalone <ul>
                if (!inStandaloneUl) {
                    result.push('<ul>');
                    inStandaloneUl = true;
                }
                result.push(`<li>${content}</li>`);
            }
            continue;
        }

        // ── Blockquotes ──
        if (trimmed.startsWith('>')) {
            result.push(`<blockquote>${trimmed.replace(/^>\s*/, '')}</blockquote>`);
            continue;
        }

        // ── Paragraphs ──
        if (trimmed && !isHeading && !trimmed.startsWith('<')) {
            result.push(`<p>${trimmed}</p>`);
        } else {
            result.push(trimmed);
        }
    }
    // ── Cleanup open tags ──
    if (inNestedUl) result.push('    </ul>');
    if (olLiOpen) result.push('  </li>');
    if (inOl) result.push('</ol>');
    if (inStandaloneUl) result.push('</ul>');

    return result.join('\n');
}

function copyArticleText() {
    if (!currentArticleData?.article_text) { toast('Chưa có nội dung', 'error'); return; }
    navigator.clipboard.writeText(currentArticleData.article_text).then(() => {
        toast('Đã copy bài viết (text)!');
    }).catch(() => {
        fallbackCopy(currentArticleData.article_text);
        toast('Đã copy bài viết (text)!');
    });
}

function copyArticleHtml() {
    if (!currentArticleData?.article_html) { toast('Chưa có HTML', 'error'); return; }
    navigator.clipboard.writeText(currentArticleData.article_html).then(() => {
        toast('Đã copy HTML! Sẵn sàng paste vào WordPress.');
    }).catch(() => {
        fallbackCopy(currentArticleData.article_html);
        toast('Đã copy HTML!');
    });
}

function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    ta.remove();
}

// Auto-load projects when page shown
document.addEventListener('DOMContentLoaded', () => {
    const projectsNav = document.querySelector('.nav-item[data-page="projects"]');
    if (projectsNav) {
        projectsNav.addEventListener('click', () => loadProjects());
    }

    // Enter key on project name input
    const prjInput = document.getElementById('newProjectName');
    if (prjInput) {
        prjInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); createProject(); }
        });
    }

    // Auto-load publish accounts when page shown
    const publishNav = document.querySelector('.nav-item[data-page="publish-accounts"]');
    if (publishNav) {
        publishNav.addEventListener('click', () => loadPublishAccounts());
    }
});

// ══════════════════════════════════════════════
// Publish Accounts Management
// ══════════════════════════════════════════════

function togglePlatformFields() {
    const platform = document.getElementById('pubPlatform').value;
    document.getElementById('wpFields').style.display = platform === 'wordpress' ? 'block' : 'none';
    document.getElementById('shopifyFields').style.display = platform === 'shopify' ? 'block' : 'none';
}

async function loadPublishAccounts() {
    const container = document.getElementById('publishAccountList');
    if (!container) return;

    try {
        const res = await fetch('/api/publish-accounts');
        const json = await res.json();
        const accounts = json.accounts || [];

        if (!accounts.length) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🚀</div>
                    <p>Chưa có tài khoản nào. Thêm tài khoản WordPress hoặc Shopify ở trên.</p>
                </div>
            `;
            return;
        }

        container.innerHTML = accounts.map(acc => {
            const platformIcon = acc.platform === 'wordpress' ? '🔵' : '🟣';
            const platformLabel = acc.platform === 'wordpress' ? 'WordPress' : 'Shopify';
            const statusClass = acc.status === 'active' ? 'positive' : 'text-dim';
            const statusLabel = acc.status === 'active' ? '✓ Active' : '○ Inactive';
            const date = new Date(acc.created_at).toLocaleDateString('vi-VN', {
                day: '2-digit', month: '2-digit', year: 'numeric'
            });

            const credInfo = acc.platform === 'wordpress'
                ? `User: ${escHtml(acc.username || '—')} · Pass: ${escHtml(acc.app_password_masked || '—')}`
                : `Token: ${escHtml(acc.api_token_masked || '—')}${acc.blog_id ? ' · Blog: ' + escHtml(acc.blog_id) : ''}`;

            // Multilingual info
            const mlPlugin = acc.multilingual_plugin || 'auto';
            const mlLabels = { 'polylang': '🌐 Polylang', 'wpml': '🌐 WPML', 'none': 'Mono-lang', 'auto': '🔍 Auto' };
            const mlLabel = mlLabels[mlPlugin] || '🔍 Auto';
            const langMap = acc.lang_mapping || {};
            const langKeys = Object.keys(langMap);
            const langInfo = langKeys.length > 0 ? `Langs: ${langKeys.join(', ')}` : '';

            return `
                <div class="publish-account-item">
                    <div class="pub-acc-icon">${platformIcon}</div>
                    <div class="pub-acc-info">
                        <div class="pub-acc-name">${escHtml(acc.name || acc.site_url)}</div>
                        <div class="pub-acc-meta">
                            <span class="pub-acc-platform">${platformLabel}</span> ·
                            <span>${escHtml(acc.site_url)}</span> ·
                            <span>${credInfo}</span> ·
                            <span>${mlLabel}</span>
                            ${langInfo ? ' · <span>' + escHtml(langInfo) + '</span>' : ''} ·
                            <span>${date}</span>
                        </div>
                    </div>
                    <div class="pub-acc-status">
                        <span class="${statusClass}">${statusLabel}</span>
                    </div>
                    <div class="pub-acc-actions">
                        <button class="btn btn-sm" onclick="testPublishAccount(${acc.id})" title="Test kết nối & nhận diện ngôn ngữ">🔌 Test</button>
                        <button class="btn btn-sm" onclick="togglePublishAccountStatus(${acc.id}, '${acc.status === 'active' ? 'inactive' : 'active'}')" title="${acc.status === 'active' ? 'Tắt' : 'Bật'}">
                            ${acc.status === 'active' ? '⏸ Tắt' : '▶ Bật'}
                        </button>
                        <button class="btn btn-sm btn-danger-sm" onclick="deletePublishAccount(${acc.id}, '${escHtml(acc.name || acc.site_url)}')" title="Xoá">🗑</button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = `<p class="negative">Lỗi: ${escHtml(err.message)}</p>`;
    }
}

async function createPublishAccount() {
    const platform = document.getElementById('pubPlatform').value;
    const name = document.getElementById('pubName').value.trim();
    const siteUrl = document.getElementById('pubSiteUrl').value.trim();

    if (!siteUrl) { toast('Vui lòng nhập Site URL', 'error'); return; }

    const data = { platform, name, site_url: siteUrl };
    data.multilingual_plugin = document.getElementById('pubMultilingualPlugin').value;

    if (platform === 'wordpress') {
        data.username = document.getElementById('pubUsername').value.trim();
        data.app_password = document.getElementById('pubAppPassword').value.trim();
        if (!data.username || !data.app_password) {
            toast('Vui lòng nhập Username và Application Password', 'error');
            return;
        }
    } else {
        data.api_token = document.getElementById('pubApiToken').value.trim();
        data.blog_id = document.getElementById('pubBlogId').value.trim();
        if (!data.api_token) {
            toast('Vui lòng nhập API Access Token', 'error');
            return;
        }
    }

    try {
        const res = await fetch('/api/publish-accounts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);

        // Clear form
        document.getElementById('pubName').value = '';
        document.getElementById('pubSiteUrl').value = '';
        document.getElementById('pubUsername').value = '';
        document.getElementById('pubAppPassword').value = '';
        document.getElementById('pubApiToken').value = '';
        document.getElementById('pubBlogId').value = '';
        document.getElementById('pubMultilingualPlugin').value = 'auto';

        toast(`Đã thêm tài khoản "${name || siteUrl}"`);
        loadPublishAccounts();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function testPublishAccount(accountId) {
    toast('Đang test kết nối & nhận diện ngôn ngữ...');
    try {
        const res = await fetch(`/api/publish-accounts/${accountId}/test`, { method: 'POST' });
        const json = await res.json();
        if (json.success) {
            toast(json.message);
            // Reload to show detected plugin info
            loadPublishAccounts();
        } else {
            toast(json.error || 'Kết nối thất bại', 'error');
        }
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function togglePublishAccountStatus(accountId, newStatus) {
    try {
        const res = await fetch(`/api/publish-accounts/${accountId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: newStatus }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast(`Đã ${newStatus === 'active' ? 'bật' : 'tắt'} tài khoản`);
        loadPublishAccounts();
    } catch (err) {
        toast(err.message, 'error');
    }
}

async function deletePublishAccount(accountId, name) {
    if (!confirm(`Xoá tài khoản "${name}"?`)) return;
    try {
        const res = await fetch(`/api/publish-accounts/${accountId}`, { method: 'DELETE' });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error);
        toast('Đã xoá tài khoản');
        loadPublishAccounts();
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ══════════════════════════════════════════════
// Publish Modal & Article Publishing
// ══════════════════════════════════════════════

let publishItemId = null;
let publishManualCategories = [];
let publishManualTags = [];

function _parseManualTerms(input) {
    return (input || '')
        .split(/[\n,]/)
        .map(s => s.trim())
        .filter(Boolean);
}

function _renderManualTermChips(containerId, terms, removeFnName) {
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!terms.length) {
        el.innerHTML = '';
        return;
    }
    el.innerHTML = terms.map((term, idx) =>
        `<span class="chip" style="display:inline-flex; align-items:center; gap:0.3rem; margin:0 0.35rem 0.35rem 0; padding:0.2rem 0.5rem; border:1px solid var(--border); border-radius:999px; background:var(--bg-secondary);">${escHtml(term)} <button type="button" class="btn btn-sm" style="padding:0 0.35rem; line-height:1.1;" onclick="${removeFnName}(${idx})">✕</button></span>`
    ).join('');
}

function addManualPublishCategory() {
    const inputEl = document.getElementById('publishCategoryManualInput');
    if (!inputEl) return;
    const items = _parseManualTerms(inputEl.value);
    if (!items.length) return;
    const seen = new Set(publishManualCategories.map(x => x.toLowerCase()));
    for (const name of items) {
        const key = name.toLowerCase();
        if (!seen.has(key)) {
            publishManualCategories.push(name);
            seen.add(key);
        }
    }
    inputEl.value = '';
    _renderManualTermChips('publishManualCategories', publishManualCategories, 'removeManualPublishCategory');
}

function removeManualPublishCategory(index) {
    publishManualCategories.splice(index, 1);
    _renderManualTermChips('publishManualCategories', publishManualCategories, 'removeManualPublishCategory');
}

function addManualPublishTag() {
    const inputEl = document.getElementById('publishTagManualInput');
    if (!inputEl) return;
    const items = _parseManualTerms(inputEl.value);
    if (!items.length) return;
    const seen = new Set(publishManualTags.map(x => x.toLowerCase()));
    for (const name of items) {
        const key = name.toLowerCase();
        if (!seen.has(key)) {
            publishManualTags.push(name);
            seen.add(key);
        }
    }
    inputEl.value = '';
    _renderManualTermChips('publishManualTags', publishManualTags, 'removeManualPublishTag');
}

function removeManualPublishTag(index) {
    publishManualTags.splice(index, 1);
    _renderManualTermChips('publishManualTags', publishManualTags, 'removeManualPublishTag');
}

function openPublishModal(itemId) {
    publishItemId = itemId;
    publishManualCategories = [];
    publishManualTags = [];
    const item = window._savedItems?.[itemId];

    // Check: has featured image?
    if (item && !item.featured_image_path) {
        toast('Chưa có ảnh đại diện. Hãy upload Featured Image trước khi publish.', 'error');
        openFeaturedImageModal(itemId);
        return;
    }

    document.getElementById('publishModalInfo').textContent =
        item ? `Đăng bài: "${truncate(item.title, 50)}"` : 'Chọn tài khoản để đăng bài:';

    document.getElementById('publishResult').innerHTML = '';
    document.getElementById('btnConfirmPublish').disabled = false;
    document.getElementById('publishSchedule').value = 'now';
    document.getElementById('scheduleTimeGroup').style.display = 'none';
    const catInput = document.getElementById('publishCategoryManualInput');
    const tagInput = document.getElementById('publishTagManualInput');
    if (catInput) catInput.value = '';
    if (tagInput) tagInput.value = '';
    _renderManualTermChips('publishManualCategories', publishManualCategories, 'removeManualPublishCategory');
    _renderManualTermChips('publishManualTags', publishManualTags, 'removeManualPublishTag');

    // Show featured image preview in publish modal
    const previewDiv = document.getElementById('publishImagePreview');
    const thumbImg = document.getElementById('publishImageThumb');
    if (item?.featured_image_path) {
        const imgName = item.featured_image_path.split(/[/\\]/).pop();
        thumbImg.src = `/uploads/featured_images/${imgName}`;
        thumbImg.alt = item.featured_image_alt || item.title || '';
        previewDiv.style.display = 'block';
    } else {
        previewDiv.style.display = 'none';
    }

    // Load publish accounts into dropdown, then update UI for platform
    loadPublishAccountDropdown().then(() => { togglePublishPlatformUI(); loadPublishCategories(); loadPublishTags(); });

    // Set language from content item
    const itemLang = item?.language || 'vi';
    const publishLangSelect = document.getElementById('publishLanguage');
    if (publishLangSelect) {
        publishLangSelect.value = itemLang;
        // If the exact value doesn't exist, try to find a close match
        if (publishLangSelect.value !== itemLang) {
            publishLangSelect.value = itemLang.split('-')[0]; // e.g. 'en-us' -> 'en'
        }
    }

    document.getElementById('publishModal').style.display = 'flex';

    if (catInput) {
        catInput.onkeydown = (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addManualPublishCategory();
            }
        };
    }
    if (tagInput) {
        tagInput.onkeydown = (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addManualPublishTag();
            }
        };
    }

    // Auto-load categories and tags when account or language changes
    document.getElementById('publishAccountSelect').onchange = () => { togglePublishPlatformUI(); loadPublishCategories(); loadPublishTags(); };
    document.getElementById('publishLanguage').onchange = () => { loadPublishCategories(); loadPublishTags(); };
}

function togglePublishPlatformUI() {
    const sel = document.getElementById('publishAccountSelect');
    const platform = sel.selectedOptions[0]?.dataset?.platform || 'wordpress';
    const isShopify = platform === 'shopify';
    document.getElementById('publishLangGroup').style.display = isShopify ? 'none' : '';
    document.getElementById('publishCatGroup').style.display = isShopify ? 'none' : '';
}

async function loadPublishCategories() {
    const accountId = document.getElementById('publishAccountSelect').value;
    const lang = document.getElementById('publishLanguage')?.value || '';
    const catSelect = document.getElementById('publishCategories');
    if (!accountId) {
        catSelect.innerHTML = '<option value="" disabled>Chọn tài khoản trước...</option>';
        return;
    }
    catSelect.innerHTML = '<option value="" disabled>Đang tải danh mục...</option>';
    try {
        const res = await fetch(`/api/publish-accounts/${accountId}/categories?lang=${encodeURIComponent(lang)}`);
        const json = await res.json();
        const cats = json.categories || [];
        if (!cats.length) {
            catSelect.innerHTML = '<option value="" disabled>Không có danh mục nào</option>';
            return;
        }
        catSelect.innerHTML = cats.map(c =>
            `<option value="${c.id}">${escHtml(c.name)} (${c.count})</option>`
        ).join('');
    } catch (e) {
        catSelect.innerHTML = '<option value="" disabled>Lỗi tải danh mục</option>';
    }
}

async function loadPublishTags() {
    const accountId = document.getElementById('publishAccountSelect').value;
    const lang = document.getElementById('publishLanguage')?.value || '';
    const tagSelect = document.getElementById('publishTags');
    if (!tagSelect) return;
    if (!accountId) {
        tagSelect.innerHTML = '<option value="" disabled>Chọn tài khoản trước...</option>';
        return;
    }
    const platform = document.getElementById('publishAccountSelect').selectedOptions[0]?.dataset?.platform || 'wordpress';
    tagSelect.innerHTML = '<option value="" disabled>Đang tải tags...</option>';
    try {
        const res = await fetch(`/api/publish-accounts/${accountId}/tags?lang=${encodeURIComponent(lang)}`);
        const json = await res.json();
        const tags = json.tags || [];
        if (!tags.length) {
            tagSelect.innerHTML = '<option value="" disabled>Không có tag nào</option>';
            return;
        }
        if (platform === 'shopify') {
            tagSelect.innerHTML = tags.map(t =>
                `<option value="${escHtml(t.name)}">${escHtml(t.name)}</option>`
            ).join('');
        } else {
            tagSelect.innerHTML = tags.map(t =>
                `<option value="${t.id}">${escHtml(t.name)} (${t.count})</option>`
            ).join('');
        }
    } catch (e) {
        tagSelect.innerHTML = '<option value="" disabled>Lỗi tải tags</option>';
    }
}
function closePublishModal() {
    document.getElementById('publishModal').style.display = 'none';
    publishItemId = null;
    publishManualCategories = [];
    publishManualTags = [];
}

function toggleScheduleTime() {
    const val = document.getElementById('publishSchedule').value;
    document.getElementById('scheduleTimeGroup').style.display = val === 'schedule' ? 'block' : 'none';
}

async function loadPublishAccountDropdown() {
    const select = document.getElementById('publishAccountSelect');
    try {
        const res = await fetch('/api/publish-accounts');
        const json = await res.json();
        const accounts = (json.accounts || []).filter(a => a.status === 'active');

        let options = '<option value="">-- Chọn tài khoản --</option>';
        for (const acc of accounts) {
            const icon = acc.platform === 'wordpress' ? '🔵' : '🟣';
            const label = acc.name || acc.site_url;
            options += `<option value="${acc.id}" data-platform="${acc.platform}">${icon} ${escHtml(label)} (${acc.platform})</option>`;
        }
        select.innerHTML = options;

        if (!accounts.length) {
            select.innerHTML = '<option value="">Chưa có tài khoản active — vào Publish Accounts để thêm</option>';
        }
    } catch (err) {
        select.innerHTML = '<option value="">Lỗi tải tài khoản</option>';
    }
}

async function confirmPublish() {
    if (!publishItemId) { toast('Không xác định được bài viết', 'error'); return; }

    const accountId = document.getElementById('publishAccountSelect').value;
    if (!accountId) { toast('Vui lòng chọn tài khoản', 'error'); return; }

    const schedule = document.getElementById('publishSchedule').value;
    let scheduleTime = null;
    if (schedule === 'schedule') {
        scheduleTime = document.getElementById('publishScheduleTime').value;
        if (!scheduleTime) { toast('Vui lòng chọn thời gian hẹn', 'error'); return; }
    }

    const btn = document.getElementById('btnConfirmPublish');
    const resultDiv = document.getElementById('publishResult');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-sm"></span> Đang đăng...';
    resultDiv.innerHTML = '';

    try {
        const publishLang = document.getElementById('publishLanguage')?.value || '';
        const platform = document.getElementById('publishAccountSelect').selectedOptions[0]?.dataset?.platform || 'wordpress';
        const catSelect = document.getElementById('publishCategories');
        const selectedCats = catSelect ? Array.from(catSelect.selectedOptions).map(o => parseInt(o.value)).filter(v => !isNaN(v)) : [];
        const tagSelect = document.getElementById('publishTags');
        let selectedTags;
        if (platform === 'shopify') {
            const selectedTagNames = tagSelect ? Array.from(tagSelect.selectedOptions).map(o => o.value).filter(v => v) : [];
            selectedTags = Array.from(new Set([...selectedTagNames, ...publishManualTags]));
        } else {
            selectedTags = tagSelect ? Array.from(tagSelect.selectedOptions).map(o => parseInt(o.value)).filter(v => !isNaN(v)) : [];
        }
        const res = await fetch(`/api/content-items/${publishItemId}/publish`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                publish_account_id: parseInt(accountId),
                schedule_time: scheduleTime,
                language: publishLang,
                categories: selectedCats,
                tags: selectedTags,
                extra_categories: publishManualCategories,
                extra_tags: platform === 'shopify' ? [] : publishManualTags,
            }),
        });
        const json = await res.json();

        if (json.success) {
            resultDiv.innerHTML = `
                <div class="publish-success">
                    <span class="positive">✅ ${escHtml(json.message)}</span>
                    ${json.published_url ? `<br><a href="${escHtml(json.published_url)}" target="_blank" rel="noopener noreferrer" class="publish-link">🔗 ${escHtml(json.published_url)}</a>` : ''}
                </div>
            `;
            toast(json.message);
            // Refresh the table
            if (currentProjectId) loadContentItems(currentProjectId);
            // Close after short delay
            setTimeout(() => closePublishModal(), 2000);
        } else {
            resultDiv.innerHTML = `<div class="publish-error"><span class="negative">❌ ${escHtml(json.error)}</span></div>`;
            toast(json.error, 'error');
        }
    } catch (err) {
        resultDiv.innerHTML = `<div class="publish-error"><span class="negative">❌ ${escHtml(err.message)}</span></div>`;
        toast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🚀 Đăng bài';
    }
}

// Close publish modal on backdrop click
document.addEventListener('click', (e) => {
    if (e.target.id === 'publishModal') closePublishModal();
    if (e.target.id === 'featuredImageModal') closeFeaturedImageModal();
    if (e.target.id === 'insertImageModal') closeInsertImageModal();
});

// ══════════════════════════════════════════════
// Insert Image into Article Editor
// ══════════════════════════════════════════════

let _selectedLibraryImage = null;

function openInsertImageModal() {
    // Reset state
    _selectedLibraryImage = null;
    document.getElementById('imgUploadFile').value = '';
    document.getElementById('imgUploadPreview').style.display = 'none';
    document.getElementById('imgUploadPlaceholder').style.display = '';
    document.getElementById('imgUploadKeyword').value = '';
    document.getElementById('imgUploadAlt').value = '';
    document.getElementById('imgUploadResult').innerHTML = '';
    document.getElementById('imgUrlInput').value = '';
    document.getElementById('imgUrlPreview').style.display = 'none';
    document.getElementById('imgUrlAlt').value = '';
    document.getElementById('imgLibrarySelected').style.display = 'none';

    // Auto-fill keyword from current item
    if (currentArticleItemId) {
        const item = window._savedItems?.[currentArticleItemId];
        if (item?.keyword) document.getElementById('imgUploadKeyword').value = item.keyword;
    }

    // Load accounts into both dropdowns
    loadImageAccountDropdowns();

    // Show upload tab by default
    switchImageTab('upload');
    document.getElementById('insertImageModal').style.display = 'flex';
}

function closeInsertImageModal() {
    document.getElementById('insertImageModal').style.display = 'none';
}

function switchImageTab(tab) {
    document.querySelectorAll('.img-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.img-tab-content').forEach(c => c.style.display = 'none');

    document.getElementById('imgTab' + tab.charAt(0).toUpperCase() + tab.slice(1)).classList.add('active');
    document.getElementById('imgContent' + tab.charAt(0).toUpperCase() + tab.slice(1)).style.display = 'block';
}

async function loadImageAccountDropdowns() {
    try {
        const res = await fetch('/api/publish-accounts');
        const json = await res.json();
        const accounts = (json.accounts || []).filter(a => a.status === 'active' && (a.platform === 'wordpress' || a.platform === 'shopify'));

        let options = '<option value="">-- Chọn publish account --</option>';
        for (const acc of accounts) {
            const icon = acc.platform === 'shopify' ? '🟣' : '🔵';
            const label = acc.platform === 'shopify' ? 'Shopify' : 'WordPress';
            options += `<option value="${acc.id}" data-platform="${acc.platform}">${icon} ${escHtml(acc.name || acc.site_url)} (${label})</option>`;
        }

        document.getElementById('imgUploadAccount').innerHTML = options;
        document.getElementById('imgLibraryAccount').innerHTML = options;

        // Auto-select if only one account
        if (accounts.length === 1) {
            document.getElementById('imgUploadAccount').value = accounts[0].id;
            document.getElementById('imgLibraryAccount').value = accounts[0].id;
        }
    } catch (err) {
        console.error('Failed to load accounts for image modal:', err);
    }
}

function onImgAccountChange() {
    // Sync both account dropdowns
    const val = document.getElementById('imgUploadAccount').value;
    document.getElementById('imgLibraryAccount').value = val;
}

function previewInsertImage(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];
        if (file.size > 10 * 1024 * 1024) {
            toast('Ảnh quá lớn. Tối đa 10MB.', 'error');
            input.value = '';
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('imgUploadPreview').src = e.target.result;
            document.getElementById('imgUploadPreview').style.display = 'block';
            document.getElementById('imgUploadPlaceholder').style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Auto-fill alt from keyword if empty
        const altInput = document.getElementById('imgUploadAlt');
        if (!altInput.value) {
            const kw = document.getElementById('imgUploadKeyword').value;
            if (kw) altInput.value = kw;
        }
    }
}

// Custom width toggle
document.addEventListener('DOMContentLoaded', () => {
    const sizeSelect = document.getElementById('imgUploadSize');
    if (sizeSelect) {
        sizeSelect.addEventListener('change', () => {
            document.getElementById('imgUploadCustomWidth').style.display =
                sizeSelect.value === 'custom' ? 'block' : 'none';
        });
    }
});

// Drag and drop support
document.addEventListener('DOMContentLoaded', () => {
    const zone = document.getElementById('imgUploadZone');
    if (!zone) return;

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            const fileInput = document.getElementById('imgUploadFile');
            fileInput.files = e.dataTransfer.files;
            previewInsertImage(fileInput);
        }
    });
});

function _buildImgTag(src, alt, align, width, mediaId) {
    let style = 'max-width:100%; height:auto;';
    if (width) style += ` width:${width}px;`;
    if (align === 'center') style += ' display:block; margin-left:auto; margin-right:auto;';
    else if (align === 'left') style += ' float:left; margin-right:1rem; margin-bottom:0.5rem;';
    else if (align === 'right') style += ' float:right; margin-left:1rem; margin-bottom:0.5rem;';

    const escapedAlt = (alt || '').replace(/"/g, '&quot;');
    const dataAttr = mediaId ? ` data-media-id="${mediaId}"` : '';
    let tag = `<img src="${src}" alt="${escapedAlt}" loading="lazy"${dataAttr} style="${style}">`;

    if (align === 'center') {
        tag = `\n<figure style="text-align:center; margin:1.5rem 0;">\n  ${tag}\n</figure>\n`;
    }
    return tag;
}

function _insertHtmlAtCursor(html) {
    const textarea = document.getElementById('articleEditCode');
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = textarea.value;

    textarea.value = text.substring(0, start) + html + text.substring(end);
    textarea.selectionStart = textarea.selectionEnd = start + html.length;
    textarea.focus();
}

async function uploadAndInsertImage() {
    const accountId = document.getElementById('imgUploadAccount').value;
    if (!accountId) { toast('Vui lòng chọn publish account', 'error'); return; }

    const uploadAccountSelect = document.getElementById('imgUploadAccount');
    const platform = uploadAccountSelect.selectedOptions[0]?.dataset?.platform || 'wordpress';
    const platformLabel = platform === 'shopify' ? 'Shopify' : 'WordPress';

    const fileInput = document.getElementById('imgUploadFile');
    if (!fileInput.files || !fileInput.files[0]) {
        toast('Vui lòng chọn file ảnh', 'error');
        return;
    }

    const keyword = document.getElementById('imgUploadKeyword').value.trim();
    const altText = document.getElementById('imgUploadAlt').value.trim();
    const align = document.getElementById('imgUploadAlign').value;
    const sizeMode = document.getElementById('imgUploadSize').value;
    const customWidth = sizeMode === 'custom' ? parseInt(document.getElementById('imgUploadWidthVal').value) : null;

    const btn = document.getElementById('btnUploadInsertImage');
    const resultDiv = document.getElementById('imgUploadResult');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-sm"></span> Đang upload lên ${platformLabel}...`;
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('alt_text', altText);
    formData.append('keyword', keyword);

    try {
        const res = await fetch(`/api/publish-accounts/${accountId}/upload-media`, {
            method: 'POST',
            body: formData,
        });
        const json = await res.json();

        if (json.success) {
            const imgTag = _buildImgTag(json.source_url, json.alt_text || altText || keyword, align, customWidth, json.media_id);
            _insertHtmlAtCursor(imgTag);

            resultDiv.innerHTML = `<span class="positive">✅ Upload thành công! (media_id: ${json.media_id}) Ảnh đã chèn vào bài viết.</span>`;
            toast(`Đã upload & chèn ảnh vào bài viết (${platformLabel})!`);

            // Reset upload form
            fileInput.value = '';
            document.getElementById('imgUploadPreview').style.display = 'none';
            document.getElementById('imgUploadPlaceholder').style.display = '';

            setTimeout(() => closeInsertImageModal(), 1200);
        } else {
            resultDiv.innerHTML = `<span class="negative">❌ ${escHtml(json.error)}</span>`;
            toast(json.error, 'error');
        }
    } catch (err) {
        resultDiv.innerHTML = `<span class="negative">❌ ${escHtml(err.message)}</span>`;
        toast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '📤 Upload & Chèn ảnh';
    }
}

// ── Media Library ──

async function loadMediaLibrary(page = 1) {
    const accountId = document.getElementById('imgLibraryAccount').value;
    const libraryAccountSelect = document.getElementById('imgLibraryAccount');
    const platform = libraryAccountSelect.selectedOptions[0]?.dataset?.platform || '';
    const platformLabel = platform === 'shopify' ? 'Shopify' : 'WordPress';

    if (!accountId) {
        document.getElementById('imgLibraryGrid').innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📚</div>
                <p>Chọn publish account để xem Media Library</p>
            </div>`;
        return;
    }

    const search = document.getElementById('imgLibrarySearch').value.trim();
    const grid = document.getElementById('imgLibraryGrid');
    grid.innerHTML = `<div class="img-library-loading"><span class="spinner-sm"></span> Đang tải Media Library (${platformLabel})...</div>`;

    try {
        let url = `/api/publish-accounts/${accountId}/media?page=${page}&per_page=20`;
        if (search) url += `&search=${encodeURIComponent(search)}`;

        const res = await fetch(url);
        const json = await res.json();

        if (!res.ok) throw new Error(json.error);

        const media = json.media || [];
        if (!media.length) {
            grid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">🖼️</div>
                    <p>Không tìm thấy ảnh nào${search ? ' cho "' + escHtml(search) + '"' : ''}</p>
                </div>`;
            document.getElementById('imgLibraryPagination').style.display = 'none';
            return;
        }

        grid.innerHTML = media.map((img, idx) => `
            <div class="img-library-item ${_selectedLibraryImage?.id === img.id ? 'selected' : ''}"
                 onclick="selectLibraryImage(${idx})"
                 title="${escHtml(img.title || img.alt_text || '')}">
                <img src="${escHtml(img.thumbnail_url || img.source_url)}" alt="${escHtml(img.alt_text || '')}" loading="lazy">
                <div class="img-library-item-info">
                    <span>${escHtml(truncate(img.title || 'Untitled', 18))}</span>
                    <small>${img.width}×${img.height}</small>
                </div>
            </div>
        `).join('');

        // Store media data for selection
        window._mediaLibraryData = media;

        // Pagination
        const pagination = document.getElementById('imgLibraryPagination');
        if (json.total_pages > 1) {
            let pagHtml = '';
            if (page > 1) pagHtml += `<button class="btn btn-sm" onclick="loadMediaLibrary(${page - 1})">← Trước</button>`;
            pagHtml += `<span style="color:var(--text-secondary); padding:0 0.5rem;">Trang ${page}/${json.total_pages} (${json.total} ảnh)</span>`;
            if (page < json.total_pages) pagHtml += `<button class="btn btn-sm" onclick="loadMediaLibrary(${page + 1})">Sau →</button>`;
            pagination.innerHTML = pagHtml;
            pagination.style.display = 'flex';
        } else {
            pagination.innerHTML = `<span style="color:var(--text-dim);">${json.total} ảnh</span>`;
            pagination.style.display = 'flex';
        }

    } catch (err) {
        grid.innerHTML = `<div class="empty-state"><p class="negative">Lỗi: ${escHtml(err.message)}</p></div>`;
    }
}

function selectLibraryImage(idx) {
    const imgData = window._mediaLibraryData?.[idx];
    if (!imgData) return;
    _selectedLibraryImage = imgData;

    // Highlight selected in grid
    document.querySelectorAll('.img-library-item').forEach((el, i) => {
        if (i === idx) el.classList.add('selected');
        else el.classList.remove('selected');
    });

    // Show selected panel  
    const selectedDiv = document.getElementById('imgLibrarySelected');
    selectedDiv.style.display = 'block';
    document.getElementById('imgLibrarySelectedImg').src = imgData.thumbnail_url || imgData.source_url;
    document.getElementById('imgLibrarySelectedName').textContent = imgData.title || 'Untitled';
    document.getElementById('imgLibrarySelectedDimension').textContent = `${imgData.width}×${imgData.height} · ${imgData.mime_type}`;
    document.getElementById('imgLibraryAlt').value = imgData.alt_text || '';
}

function insertImageFromLibrary() {
    if (!_selectedLibraryImage) { toast('Vui lòng chọn ảnh từ Media Library', 'error'); return; }

    const altText = document.getElementById('imgLibraryAlt').value.trim();
    const align = document.getElementById('imgLibraryAlign').value;
    const imgTag = _buildImgTag(_selectedLibraryImage.source_url, altText, align, null, _selectedLibraryImage.id);

    _insertHtmlAtCursor(imgTag);
    toast(`Đã chèn ảnh từ Media Library! (media_id: ${_selectedLibraryImage.id})`);
    closeInsertImageModal();
}

// ── Image URL ──

function previewUrlImage() {
    const url = document.getElementById('imgUrlInput').value.trim();
    const preview = document.getElementById('imgUrlPreview');
    const previewImg = document.getElementById('imgUrlPreviewImg');

    if (url && /^https?:\/\/.+\.(jpg|jpeg|png|gif|webp|svg|bmp)/i.test(url)) {
        previewImg.src = url;
        preview.style.display = 'block';
        previewImg.onerror = () => { preview.style.display = 'none'; };
    } else {
        preview.style.display = 'none';
    }
}

function insertImageFromUrl() {
    const url = document.getElementById('imgUrlInput').value.trim();
    if (!url) { toast('Vui lòng nhập URL ảnh', 'error'); return; }

    const altText = document.getElementById('imgUrlAlt').value.trim();
    const align = document.getElementById('imgUrlAlign').value;
    const imgTag = _buildImgTag(url, altText, align, null);

    _insertHtmlAtCursor(imgTag);
    toast('Đã chèn ảnh từ URL!');
    closeInsertImageModal();
}

// ══════════════════════════════════════════════
// Featured Image Modal
// ══════════════════════════════════════════════

let featuredImageItemId = null;

function openFeaturedImageModal(itemId) {
    featuredImageItemId = itemId;
    const item = window._savedItems?.[itemId];

    // Reset modal
    document.getElementById('featuredImageFile').value = '';
    document.getElementById('featuredImageAlt').value = item?.featured_image_alt || '';
    document.getElementById('featuredImageResult').innerHTML = '';
    document.getElementById('featuredImageImg').style.display = 'none';
    document.getElementById('featuredImageEmpty').style.display = 'block';
    document.getElementById('btnRemoveFeaturedImage').style.display = 'none';

    // Load existing image if any
    if (item?.featured_image_path) {
        const imgName = item.featured_image_path.split(/[/\\]/).pop();
        const imgUrl = `/uploads/featured_images/${imgName}`;
        document.getElementById('featuredImageImg').src = imgUrl;
        document.getElementById('featuredImageImg').style.display = 'block';
        document.getElementById('featuredImageEmpty').style.display = 'none';
        document.getElementById('btnRemoveFeaturedImage').style.display = 'inline-flex';
        document.getElementById('featuredImageAlt').value = item.featured_image_alt || '';
    }

    document.getElementById('featuredImageModal').style.display = 'flex';
}

function closeFeaturedImageModal() {
    document.getElementById('featuredImageModal').style.display = 'none';
    featuredImageItemId = null;
}

function previewFeaturedImage(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];

        // Validate size
        if (file.size > 10 * 1024 * 1024) {
            toast('Ảnh quá lớn. Tối đa 10MB.', 'error');
            input.value = '';
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('featuredImageImg').src = e.target.result;
            document.getElementById('featuredImageImg').style.display = 'block';
            document.getElementById('featuredImageEmpty').style.display = 'none';
        };
        reader.readAsDataURL(file);

        // Auto-generate alt from title if empty
        const altInput = document.getElementById('featuredImageAlt');
        if (!altInput.value && featuredImageItemId) {
            const item = window._savedItems?.[featuredImageItemId];
            if (item?.title) altInput.value = item.title;
        }
    }
}

async function saveFeaturedImage() {
    if (!featuredImageItemId) return;

    const fileInput = document.getElementById('featuredImageFile');
    const altText = document.getElementById('featuredImageAlt').value.trim();
    const resultDiv = document.getElementById('featuredImageResult');
    const btn = document.getElementById('btnSaveFeaturedImage');

    if (!fileInput.files || !fileInput.files[0]) {
        // If no new file but alt text changed, update alt text only
        if (altText && window._savedItems?.[featuredImageItemId]?.featured_image_path) {
            try {
                btn.disabled = true;
                btn.textContent = '⏳ Đang lưu...';
                const res = await fetch(`/api/content-items/${featuredImageItemId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ featured_image_alt: altText }),
                });
                const json = await res.json();
                if (!res.ok) throw new Error(json.error);
                toast('Đã cập nhật alt text');
                // Update local cache
                if (window._savedItems[featuredImageItemId]) {
                    window._savedItems[featuredImageItemId].featured_image_alt = altText;
                }
                closeFeaturedImageModal();
                if (currentProjectId) loadContentItems(currentProjectId);
            } catch (err) {
                resultDiv.innerHTML = `<span class="negative">❌ ${escHtml(err.message)}</span>`;
            } finally {
                btn.disabled = false;
                btn.textContent = '💾 Lưu ảnh';
            }
            return;
        }
        toast('Vui lòng chọn file ảnh', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('alt_text', altText);

    btn.disabled = true;
    btn.textContent = '⏳ Đang upload...';
    resultDiv.innerHTML = '';

    try {
        const res = await fetch(`/api/content-items/${featuredImageItemId}/featured-image`, {
            method: 'POST',
            body: formData,
        });
        const json = await res.json();

        if (json.success) {
            toast(json.message);
            resultDiv.innerHTML = `<span class="positive">✅ ${escHtml(json.message)}</span>`;
            // Update local cache
            if (window._savedItems[featuredImageItemId]) {
                window._savedItems[featuredImageItemId].featured_image_path = json.featured_image_path;
                window._savedItems[featuredImageItemId].featured_image_alt = json.featured_image_alt;
            }
            // Refresh table
            if (currentProjectId) loadContentItems(currentProjectId);
            setTimeout(() => closeFeaturedImageModal(), 1000);
        } else {
            resultDiv.innerHTML = `<span class="negative">❌ ${escHtml(json.error)}</span>`;
            toast(json.error, 'error');
        }
    } catch (err) {
        resultDiv.innerHTML = `<span class="negative">❌ ${escHtml(err.message)}</span>`;
        toast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '💾 Lưu ảnh';
    }
}

async function removeFeaturedImage() {
    if (!featuredImageItemId) return;
    if (!confirm('Xoá ảnh đại diện?')) return;

    try {
        const res = await fetch(`/api/content-items/${featuredImageItemId}/featured-image`, {
            method: 'DELETE',
        });
        const json = await res.json();
        if (json.success) {
            toast('Đã xoá ảnh đại diện');
            // Update local cache
            if (window._savedItems[featuredImageItemId]) {
                window._savedItems[featuredImageItemId].featured_image_path = '';
                window._savedItems[featuredImageItemId].featured_image_alt = '';
            }
            // Refresh table
            if (currentProjectId) loadContentItems(currentProjectId);
            closeFeaturedImageModal();
        } else {
            toast(json.error || 'Lỗi xoá ảnh', 'error');
        }
    } catch (err) {
        toast(err.message, 'error');
    }
}
