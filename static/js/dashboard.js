/**
 * Dashboard JavaScript for Smart Lock System
 * Real-time updates, charts, and interactive features
 */

class SmartLockDashboard {
    constructor() {
        this.updateInterval = null;
        this.charts = {};
        this.currentStats = {};
        this.currentReportsOwner = 'overall';
        this.reportsRefreshInterval = null;
        
        this.init();
    }
    
    init() {
        console.log('📊 Initializing Smart Lock Dashboard...');
        
        // Load initial data
        this.loadSystemStats();
        this.loadRecentEvents();
        this.loadCharts();
        
        // Set up auto-refresh
        this.startAutoRefresh();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initialize real-time updates
        this.initRealTimeUpdates();
    }
    
    setupEventListeners() {
        // Refresh buttons
        document.querySelectorAll('.refresh-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                this.refreshAll();
                this.showToast('Data refreshed', 'success');
            });
        });
        
        // Export buttons
        document.getElementById('exportCsv')?.addEventListener('click', () => this.exportData('csv'));
        document.getElementById('exportJson')?.addEventListener('click', () => this.exportData('json'));
        
        // Filter controls
        document.getElementById('timeFilter')?.addEventListener('change', (e) => {
            this.applyTimeFilter(e.target.value);
        });
        
        document.getElementById('decisionFilter')?.addEventListener('change', (e) => {
            this.applyDecisionFilter(e.target.value);
        });
        
        // Chart type toggles
        document.querySelectorAll('.chart-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                const chartType = e.target.dataset.chart;
                this.toggleChart(chartType);
            });
        });
        
        // Real-time toggle
        document.getElementById('realtimeToggle')?.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        });

        // Reports modal close
        document.getElementById('closeReports')?.addEventListener('click', () => this.closeReportsModal());
    }

    async openReportsModal() {
        const modal = document.getElementById('reportsModal');
        if (!modal) return;
        modal.style.display = 'block';
        
        // Set current owner to overall
        this.currentReportsOwner = 'overall';
        
        // Setup owner filter buttons
        document.querySelectorAll('.ownerFilterBtn').forEach(btn => {
            btn.style.background = btn.dataset.owner === 'overall' ? 'rgba(0,180,219,0.15)' : 'transparent';
            btn.style.borderColor = btn.dataset.owner === 'overall' ? 'rgba(0,180,219,0.3)' : 'rgba(255,255,255,0.1)';
            btn.addEventListener('click', async (e) => {
                e.preventDefault();
                this.currentReportsOwner = btn.dataset.owner;
                
                // Update button styles
                document.querySelectorAll('.ownerFilterBtn').forEach(b => {
                    b.style.background = b.dataset.owner === this.currentReportsOwner ? 'rgba(0,180,219,0.15)' : 'transparent';
                    b.style.borderColor = b.dataset.owner === this.currentReportsOwner ? 'rgba(0,180,219,0.3)' : 'rgba(255,255,255,0.1)';
                    b.style.color = b.dataset.owner === this.currentReportsOwner ? '#00b4db' : '#ecf0f1';
                });
                
                // Fetch and render with new owner filter
                await this.fetchAndRenderReports(this.currentReportsOwner);
            });
        });
        
        // Setup refresh button
        document.getElementById('refreshReportsBtn')?.addEventListener('click', async (e) => {
            e.preventDefault();
            await this.fetchAndRenderReports(this.currentReportsOwner);
        });
        
        // Setup auto-refresh toggle
        const autoRefreshToggle = document.getElementById('reportsAutoRefresh');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startReportsAutoRefresh();
                } else {
                    this.stopReportsAutoRefresh();
                }
            });
            if (autoRefreshToggle.checked) {
                this.startReportsAutoRefresh();
            }
        }
        
        // Fetch and render reports
        await this.fetchAndRenderReports('overall');
    }

    closeReportsModal() {
        const modal = document.getElementById('reportsModal');
        if (!modal) return;
        modal.style.display = 'none';
        this.stopReportsAutoRefresh();
    }

    startReportsAutoRefresh() {
        this.reportsRefreshInterval = setInterval(async () => {
            if (document.getElementById('reportsModal')?.style.display === 'block') {
                await this.fetchAndRenderReports(this.currentReportsOwner || 'overall');
            }
        }, 10000);
    }

    stopReportsAutoRefresh() {
        if (this.reportsRefreshInterval) {
            clearInterval(this.reportsRefreshInterval);
        }
    }

    async fetchAndRenderReports(owner = 'overall') {
        try {
            const url = `/reports?owner=${encodeURIComponent(owner)}`;
            const resp = await fetch(url);
            const data = await resp.json();
            if (data.error) {
                console.error('Reports error:', data.error);
                return;
            }
            
            // Update stats
            document.getElementById('statsTotal').textContent = data.total_events || 0;
            document.getElementById('statsSuccess').textContent = Math.round(data.success_rate || 0) + '%';
            document.getElementById('statsAvgFace').textContent = (data.score_stats?.avg_face || 0).toFixed(2);
            document.getElementById('statsAvgVoice').textContent = (data.score_stats?.avg_voice || 0).toFixed(2);
            document.getElementById('reportsLastUpdated').textContent = new Date().toLocaleTimeString();
            
            // Normalize data to avoid blank charts
            const trends = Array.isArray(data.trends) ? data.trends : this.buildEmptyTrends(14);
            const decisionDist = data.decision_dist || { ALLOW: 0, DENY: 0, LOCKOUT: 0 };
            const actionDist = data.action_dist || { IN: 0, OUT: 0 };

            // Render all charts
            this.renderHeatmap(data.heatmap || data.calendar);
            this.renderTrendChart(trends);
            this.renderDecisionChart(decisionDist);
            this.renderActionChart(actionDist);
            this.renderRiskChart(trends);
        } catch (err) {
            console.error('Error fetching reports:', err);
        }
    }

    // Calendar-style month × day heatmap (replaces old hourly heatmap)
    renderCalendarHeatmap(calendarData, containerId = 'heatmapContainer') {
        if (calendarData && typeof calendarData === 'object' && !Array.isArray(calendarData)) {
            this.renderHeatmap(calendarData, containerId);
            return;
        }

        const container = document.getElementById(containerId);
        if (!container) return;

        if (!Array.isArray(calendarData) && calendarData && typeof calendarData === 'object') {
            calendarData = this.normalizeCalendarDataFromHeatmap(calendarData);
        }

        container.innerHTML = '';

        if (!calendarData || calendarData.length === 0) {
            container.innerHTML = '<p style="color:#888; margin:8px 0;">No data available</p>';
            return;
        }

        // Build lookup map for quick access
        const dataMap = {};
        calendarData.forEach(d => { dataMap[d.date] = d.count; });

        const today = new Date();
        const year = today.getFullYear();
        const month = today.getMonth(); // current month

        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);

        const startDayOfWeek = firstDay.getDay(); // 0 = Sun
        const totalDays = lastDay.getDate();

        // Determine max for intensity scaling
        const maxCount = Math.max(...calendarData.map(d => d.count), 1);

        // Container header (month/year)
        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        header.style.marginBottom = '8px';
        header.style.color = '#bdc3c7';
        header.innerHTML = `<strong>${firstDay.toLocaleString(undefined, { month: 'long', year: 'numeric' })}</strong>`;
        container.appendChild(header);

        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'repeat(7, 1fr)';
        grid.style.gap = '6px';
        grid.style.fontSize = '12px';

        // Weekday headers
        const weekdays = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
        weekdays.forEach(w => {
            const h = document.createElement('div');
            h.textContent = w;
            h.style.textAlign = 'center';
            h.style.color = '#9aa6b2';
            h.style.fontWeight = '600';
            grid.appendChild(h);
        });

        // Empty cells before the first day
        for (let i = 0; i < startDayOfWeek; i++) {
            const empty = document.createElement('div');
            grid.appendChild(empty);
        }

        // Day cells
        for (let day = 1; day <= totalDays; day++) {
            const dateObj = new Date(year, month, day);
            const iso = dateObj.toISOString().split('T')[0];
            const count = dataMap[iso] || 0;
            const intensity = count / maxCount;

            const cell = document.createElement('div');
            cell.textContent = day;
            cell.style.padding = '8px 6px';
            cell.style.borderRadius = '6px';
            cell.style.textAlign = 'center';
            cell.style.cursor = 'default';
            cell.style.background = `rgba(0,180,219,${Math.min(1, intensity * 1.2)})`;
            cell.style.color = intensity > 0.5 ? '#ffffff' : '#c7d2da';
            cell.title = `${iso} — ${count} events`;

            // show small badge when there are events
            if (count > 0) {
                const badge = document.createElement('div');
                badge.style.fontSize = '10px';
                badge.style.opacity = '0.85';
                badge.style.marginTop = '6px';
                badge.textContent = count;
                badge.style.color = intensity > 0.5 ? '#fff' : '#aab7bf';
                cell.appendChild(badge);
            }

            grid.appendChild(cell);
        }

        container.appendChild(grid);
    }

    normalizeCalendarDataFromHeatmap(heatmapData) {
        if (!heatmapData || typeof heatmapData !== 'object') return [];
        const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
        const today = new Date();
        const year = today.getFullYear();
        const month = today.getMonth();
        const totalDays = new Date(year, month + 1, 0).getDate();
        const out = [];

        for (let day = 1; day <= totalDays; day++) {
            const dateObj = new Date(year, month, day);
            const weekday = days[dateObj.getDay()];
            const hourly = heatmapData[weekday] || [];
            const count = Array.isArray(hourly) ? hourly.reduce((a, b) => a + (Number(b) || 0), 0) : 0;
            out.push({ date: dateObj.toISOString().slice(0, 10), count });
        }
        return out;
    }

    // Day (rows) x Hour (columns) heatmap for reports, aggregated across all months
    renderHeatmap(heatmap, containerId = 'heatmapContainer') {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        if (!heatmap || typeof heatmap !== 'object') {
            container.innerHTML = '<p style="color:#888; margin:8px 0;">No data available</p>';
            return;
        }

        const table = document.createElement('table');
        table.style.width = '100%';
        table.style.borderCollapse = 'separate';
        table.style.borderSpacing = '10px';
        table.style.tableLayout = 'fixed';
        table.style.fontSize = '15px';

        const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
        const labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

        let globalMax = 1;
        days.forEach((day) => {
            const values = Array.isArray(heatmap[day]) ? heatmap[day] : [];
            values.forEach((v) => {
                const n = Number(v) || 0;
                if (n > globalMax) globalMax = n;
            });
        });

        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        const th0 = document.createElement('th');
        th0.textContent = 'Day';
        th0.style.padding = '12px 10px';
        th0.style.color = '#bdc3c7';
        th0.style.fontSize = '14px';
        th0.style.width = '9%';
        headRow.appendChild(th0);

        for (let h = 0; h < 24; h++) {
            const th = document.createElement('th');
            th.textContent = String(h).padStart(2, '0');
            th.style.padding = '10px 8px';
            th.style.color = '#bdc3c7';
            th.style.fontSize = '12px';
            headRow.appendChild(th);
        }
        thead.appendChild(headRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        days.forEach((day, idx) => {
            const row = document.createElement('tr');
            const labelCell = document.createElement('td');
            labelCell.textContent = labels[idx];
            labelCell.style.padding = '12px 10px';
            labelCell.style.color = '#ecf0f1';
            labelCell.style.fontWeight = 'bold';
            labelCell.style.fontSize = '15px';
            labelCell.style.width = '9%';
            row.appendChild(labelCell);

            const values = Array.isArray(heatmap[day]) ? heatmap[day] : Array(24).fill(0);
            for (let h = 0; h < 24; h++) {
                const v = Number(values[h]) || 0;
                const td = document.createElement('td');
                td.style.padding = '16px 0';
                td.style.textAlign = 'center';
                td.style.fontSize = '13px';
                td.style.fontWeight = '600';
                td.style.borderRadius = '10px';
                td.style.boxShadow = 'inset 0 0 0 1px rgba(255,255,255,0.06)';

                const intensity = v / globalMax;
                // Blue heatmap: denser = deeper blue, zero = dark cell
                if (v === 0) {
                    td.style.background = 'rgba(255,255,255,0.04)';
                    td.style.color = 'transparent';
                } else {
                    td.style.background = `rgba(0,140,255,${Math.min(1, 0.15 + intensity * 0.85)})`;
                    td.style.color = intensity > 0.55 ? '#eef6ff' : '#cfe6ff';
                }
                td.textContent = v > 0 ? String(v) : '';
                td.title = `${day} ${String(h).padStart(2, '0')}:00 - ${v} events`;
                row.appendChild(td);
            }
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        container.appendChild(table);
    }

    // Date (rows) x Hour (columns) heatmap for last N days
    renderDateHourHeatmap(heatmap, containerId = 'heatmapContainer') {
        const container = document.getElementById(containerId);
        if (!container) return;
        container.innerHTML = '';

        if (!heatmap || typeof heatmap !== 'object') {
            container.innerHTML = '<p style="color:#888; margin:8px 0;">No data available</p>';
            return;
        }

        const isDateMap = !Array.isArray(heatmap) && Object.keys(heatmap).some(k => /^\d{4}-\d{2}-\d{2}$/.test(k));
        if (!isDateMap) {
            // Fallback to existing weekday/hour heatmap
            this.renderHeatmap(heatmap, containerId);
            return;
        }

        const dates = Object.keys(heatmap).sort();
        if (dates.length === 0) {
            container.innerHTML = '<p style="color:#888; margin:8px 0;">No data available</p>';
            return;
        }

        let globalMax = 1;
        dates.forEach((d) => {
            const values = Array.isArray(heatmap[d]) ? heatmap[d] : [];
            values.forEach((v) => {
                const n = Number(v) || 0;
                if (n > globalMax) globalMax = n;
            });
        });

        const table = document.createElement('table');
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';
        table.style.fontSize = '11px';

        const thead = document.createElement('thead');
        const headRow = document.createElement('tr');
        const th0 = document.createElement('th');
        th0.textContent = 'Date';
        th0.style.padding = '4px';
        th0.style.color = '#bdc3c7';
        headRow.appendChild(th0);

        for (let h = 0; h < 24; h++) {
            const th = document.createElement('th');
            th.textContent = String(h).padStart(2, '0');
            th.style.padding = '3px';
            th.style.color = '#bdc3c7';
            th.style.fontSize = '9px';
            headRow.appendChild(th);
        }
        thead.appendChild(headRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        dates.forEach((dateStr) => {
            const row = document.createElement('tr');
            const labelCell = document.createElement('td');
            const labelDate = new Date(dateStr + 'T00:00:00');
            const dayLabel = labelDate.toLocaleDateString(undefined, { weekday: 'short' });
            const shortLabel = `${dayLabel} ${labelDate.toLocaleDateString(undefined, { month: 'short', day: '2-digit' })}`;
            labelCell.textContent = shortLabel;
            labelCell.title = dateStr;
            labelCell.style.padding = '4px';
            labelCell.style.color = '#ecf0f1';
            labelCell.style.fontWeight = 'bold';
            labelCell.style.fontSize = '10px';
            row.appendChild(labelCell);

            const values = Array.isArray(heatmap[dateStr]) ? heatmap[dateStr] : Array(24).fill(0);
            for (let h = 0; h < 24; h++) {
                const v = Number(values[h]) || 0;
                const td = document.createElement('td');
                td.style.padding = '3px';
                td.style.textAlign = 'center';
                td.style.fontSize = '9px';
                td.style.borderRadius = '3px';

                const intensity = v / globalMax;
                td.style.background = `rgba(0,180,219,${Math.min(1, intensity * 1.2)})`;
                td.style.color = intensity > 0.5 ? '#ecf0f1' : '#bdc3c7';
                td.textContent = v > 0 ? String(v) : '';
                td.title = `${dateStr} ${String(h).padStart(2, '0')}:00 - ${v} events`;
                row.appendChild(td);
            }
            tbody.appendChild(row);
        });
        table.appendChild(tbody);
        container.appendChild(table);
    }

    renderTrendChart(trends) {
        const ctx = document.getElementById('reportsTrendChart');
        if (!ctx || typeof Chart === 'undefined') return;
        
        const labels = trends.map(t => t.day);
        const totals = trends.map(t => t.total || 0);
        const highs = trends.map(t => t.high || 0);
        const mediums = trends.map(t => t.medium || 0);
        const lows = trends.map(t => t.low || 0);

        if (this.charts.reportsTrend) {
            this.charts.reportsTrend.data.labels = labels;
            this.charts.reportsTrend.data.datasets[0].data = totals;
            this.charts.reportsTrend.data.datasets[1].data = highs;
            this.charts.reportsTrend.data.datasets[2].data = mediums;
            this.charts.reportsTrend.data.datasets[3].data = lows;
            this.charts.reportsTrend.update();
            return;
        }

        this.charts.reportsTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Total', data: totals, borderColor: '#3498db', backgroundColor: 'rgba(52,152,219,0.1)', fill: true, tension: 0.3, pointBackgroundColor: '#3498db', pointBorderColor: '#fff' },
                    { label: 'High Risk', data: highs, borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.05)', fill: true, tension: 0.3, pointBackgroundColor: '#e74c3c', pointBorderColor: '#fff' },
                    { label: 'Medium Risk', data: mediums, borderColor: '#f39c12', backgroundColor: 'rgba(243,156,18,0.05)', fill: true, tension: 0.3, pointBackgroundColor: '#f39c12', pointBorderColor: '#fff' },
                    { label: 'Low Risk', data: lows, borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.05)', fill: true, tension: 0.3, pointBackgroundColor: '#2ecc71', pointBorderColor: '#fff' }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { labels: { color: '#bdc3c7' } } },
                scales: {
                    y: { ticks: { color: '#bdc3c7' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                    x: { ticks: { color: '#bdc3c7' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }

    buildEmptyTrends(days = 14) {
        const out = [];
        const today = new Date();
        for (let i = days - 1; i >= 0; i--) {
            const d = new Date(today);
            d.setDate(today.getDate() - i);
            out.push({
                day: d.toISOString().slice(0, 10),
                total: 0,
                high: 0,
                medium: 0,
                low: 0
            });
        }
        return out;
    }

    renderDecisionChart(decisionDist) {
        const ctx = document.getElementById('reportsDecisionChart');
        if (!ctx || typeof Chart === 'undefined') return;

        const data = {
            'ALLOW': decisionDist.ALLOW || 0,
            'DENY': decisionDist.DENY || 0,
            'LOCKOUT': decisionDist.LOCKOUT || 0
        };

        if (this.charts.reportsDecision) {
            this.charts.reportsDecision.data.datasets[0].data = Object.values(data);
            this.charts.reportsDecision.update();
            return;
        }

        this.charts.reportsDecision = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    data: Object.values(data),
                    backgroundColor: ['#2ecc71', '#e74c3c', '#9b59b6'],
                    borderColor: 'rgba(10,10,10,0.95)',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { labels: { color: '#bdc3c7' } },
                    tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.parsed}` } }
                }
            }
        });
    }

    renderActionChart(actionDist) {
        const ctx = document.getElementById('reportsActionChart');
        if (!ctx || typeof Chart === 'undefined') return;

        const data = {
            'IN': actionDist.IN || 0,
            'OUT': actionDist.OUT || 0
        };

        if (this.charts.reportsAction) {
            this.charts.reportsAction.data.datasets[0].data = Object.values(data);
            this.charts.reportsAction.update();
            return;
        }

        this.charts.reportsAction = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(data),
                datasets: [{
                    data: Object.values(data),
                    backgroundColor: ['#3498db', '#e67e22'],
                    borderColor: 'rgba(10,10,10,0.95)',
                    borderWidth: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { labels: { color: '#bdc3c7' } },
                    tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.parsed}` } }
                }
            }
        });
    }

    renderRiskChart(trends) {
        const ctx = document.getElementById('reportsRiskChart');
        if (!ctx || typeof Chart === 'undefined') return;

        const totalHigh = trends.reduce((sum, t) => sum + (t.high || 0), 0);
        const totalMedium = trends.reduce((sum, t) => sum + (t.medium || 0), 0);
        const totalLow = trends.reduce((sum, t) => sum + (t.low || 0), 0);
        const total = totalHigh + totalMedium + totalLow;

        if (this.charts.reportsRisk) {
            this.charts.reportsRisk.data.datasets[0].data = [totalHigh, totalMedium, totalLow];
            this.charts.reportsRisk.update();
            return;
        }

        this.charts.reportsRisk = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [{
                    label: 'Event Count',
                    data: [totalHigh, totalMedium, totalLow],
                    backgroundColor: ['#e74c3c', '#f39c12', '#2ecc71'],
                    borderColor: ['#c0392b', '#d68910', '#27ae60'],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                indexAxis: 'x',
                plugins: {
                    legend: { display: true, labels: { color: '#bdc3c7' } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                if (!total) return `${ctx.parsed.y} events`;
                                return `${ctx.parsed.y} events (${((ctx.parsed.y / total) * 100).toFixed(1)}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: { ticks: { color: '#bdc3c7' }, grid: { color: 'rgba(255,255,255,0.05)' }, beginAtZero: true },
                    x: { ticks: { color: '#bdc3c7' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    }
    
    initRealTimeUpdates() {
        // Set up WebSocket for real-time updates (if supported)
        if (window.WebSocket) {
            this.initWebSocket();
        } else {
            console.log('WebSocket not supported, using polling');
        }
        
        // Set up service worker for push notifications
        if ('serviceWorker' in navigator && 'PushManager' in window) {
            this.registerServiceWorker();
        }
    }
    
    initWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('🔗 WebSocket connected');
                this.showToast('Real-time updates enabled', 'success');
            };
            
            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.ws.onclose = () => {
                console.log('🔌 WebSocket disconnected');
                // Attempt to reconnect
                setTimeout(() => this.initWebSocket(), 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'new_event':
                this.addNewEvent(data.event);
                this.updateStats(data.stats);
                this.updateCharts();
                this.showNotification('New access attempt', data.event.description);
                break;
                
            case 'lockout':
                this.showLockoutAlert(data.reason);
                break;
                
            case 'stats_update':
                this.updateStats(data.stats);
                break;
                
            case 'system_alert':
                this.showSystemAlert(data.message, data.level);
                break;
        }
    }
    
    registerServiceWorker() {
        navigator.serviceWorker.register('/service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registered:', registration);
                
                // Request notification permission
                if (Notification.permission === 'default') {
                    Notification.requestPermission();
                }
            })
            .catch(error => {
                console.error('ServiceWorker registration failed:', error);
            });
    }
    
    async loadSystemStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            this.currentStats = data;
            this.updateStatsDisplay(data);
            
        } catch (error) {
            console.error('Error loading system stats:', error);
            this.showToast('Failed to load statistics', 'error');
        }
    }
    
    updateStatsDisplay(stats) {
        // Update summary cards
        this.updateElement('totalAttempts', stats.total_attempts || 0);
        this.updateElement('successRate', `${((stats.success_rate || 0) * 100).toFixed(1)}%`);
        this.updateElement('avgResponseTime', `${stats.avg_response_time || 0}ms`);
        this.updateElement('currentLockouts', stats.active_lockouts || 0);
        
        // Update detailed stats
        if (stats.detailed) {
            this.updateElement('todayAttempts', stats.detailed.today || 0);
            this.updateElement('weekAttempts', stats.detailed.week || 0);
            this.updateElement('monthAttempts', stats.detailed.month || 0);
            
            // Update score averages
            this.updateElement('avgFaceScore', (stats.detailed.avg_face_score || 0).toFixed(3));
            this.updateElement('avgVoiceScore', (stats.detailed.avg_voice_score || 0).toFixed(3));
            this.updateElement('avgFinalScore', (stats.detailed.avg_final_score || 0).toFixed(3));
        }
        
        // Update AI status
        const aiStatus = document.getElementById('aiStatus');
        if (aiStatus) {
            aiStatus.textContent = stats.ai_connected ? '🟢 Connected' : '🔴 Disconnected';
            aiStatus.className = stats.ai_connected ? 'status-connected' : 'status-disconnected';
        }
    }
    
    async loadRecentEvents(limit = 20) {
        try {
            const response = await fetch(`/api/events?limit=${limit}`);
            const data = await response.json();
            
            if (data.events) {
                this.updateEventsTable(data.events);
                this.updateEventsChart(data.events);
            }
            
        } catch (error) {
            console.error('Error loading recent events:', error);
        }
    }
    
    updateEventsTable(events) {
        const tableBody = document.getElementById('eventsTableBody');
        if (!tableBody) return;
        
        tableBody.innerHTML = '';
        
        events.forEach(event => {
            const row = document.createElement('tr');
            row.className = `event-row decision-${event.decision.toLowerCase()}`;
            
            // Format time
            const eventTime = new Date(event.time);
            const timeStr = eventTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const dateStr = eventTime.toLocaleDateString();
            
            // Create risk badge
            const riskBadge = `<span class="risk-badge risk-${event.risk_level.toLowerCase()}">${event.risk_level}</span>`;
            
            // Create scores display
            const scores = `PIN: ${event.pin_valid ? '✓' : '✗'} | Face: ${event.face_score?.toFixed(2) || 'N/A'} | Voice: ${event.voice_score?.toFixed(2) || 'N/A'}`;
            
            row.innerHTML = `
                <td>${dateStr} ${timeStr}</td>
                <td>${event.action}</td>
                <td>${event.pin_valid ? '✓' : '✗'}</td>
                <td>${event.face_score?.toFixed(2) || 'N/A'}</td>
                <td>${event.voice_score?.toFixed(2) || 'N/A'}</td>
                <td>${event.final_score?.toFixed(2) || 'N/A'}</td>
                <td>
                    <span class="decision-badge decision-${event.decision.toLowerCase()}">
                        ${this.getDecisionIcon(event.decision)} ${event.decision}
                    </span>
                </td>
                <td>${riskBadge}</td>
                <td>${event.explanation || ''}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    addNewEvent(event) {
        // Add to table
        const tableBody = document.getElementById('eventsTableBody');
        if (tableBody) {
            const existingRow = tableBody.querySelector(`[data-event-id="${event.id}"]`);
            if (!existingRow) {
                this.updateEventsTable([event, ...this.currentEvents || []]);
                this.currentEvents = [event, ...(this.currentEvents || []).slice(0, 19)];
            }
        }
        
        // Update charts
        this.updateCharts();
        
        // Show notification
        if (event.decision === 'LOCKOUT') {
            this.showLockoutAlert(event.explanation);
        }
    }
    
    loadCharts() {
        // Initialize Chart.js charts if available
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded, charts disabled');
            return;
        }
        
        // Decision distribution chart
        this.initDecisionChart();
        
        // Score trends chart
        this.initScoreTrendsChart();
        
        // Time distribution chart
        this.initTimeDistributionChart();
        
        // Risk analysis chart
        this.initRiskAnalysisChart();

        // Dashboard embedded heatmap removed per UI requirement
    }
    
    initDecisionChart() {
        const ctx = document.getElementById('decisionChart');
        if (!ctx) return;
        
        this.charts.decision = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Allowed', 'Denied', 'Lockout'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        '#2ecc71', // Green for allowed
                        '#e74c3c', // Red for denied
                        '#9b59b6'  // Purple for lockout
                    ],
                    borderWidth: 2,
                    borderColor: '#1a1a2e'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ecf0f1',
                            font: {
                                size: 12
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: 'Decision Distribution',
                        color: '#ecf0f1',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            }
        });
    }
    
    initScoreTrendsChart() {
        const ctx = document.getElementById('scoreTrendsChart');
        if (!ctx) return;
        
        this.charts.scores = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Face Score',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Voice Score',
                        data: [],
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230, 126, 34, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Final Score',
                        data: [],
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ecf0f1'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Score Trends',
                        color: '#ecf0f1'
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#bdc3c7'
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#bdc3c7'
                        }
                    }
                }
            }
        });
    }
    
    initTimeDistributionChart() {
        const ctx = document.getElementById('timeDistributionChart');
        if (!ctx) return;
        
        this.charts.time = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                        '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                datasets: [{
                    label: 'Access Attempts',
                    data: new Array(24).fill(0),
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: '#3498db',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#ecf0f1'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Access Time Distribution',
                        color: '#ecf0f1'
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#bdc3c7'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#bdc3c7'
                        }
                    }
                }
            }
        });
    }
    
    initRiskAnalysisChart() {
        const ctx = document.getElementById('riskAnalysisChart');
        if (!ctx) return;
        
        this.charts.risk = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['PIN Validity', 'Face Match', 'Voice Match', 'Time Pattern', 'Behavior Score'],
                datasets: [{
                    label: 'Risk Factors',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(231, 76, 60, 0.2)',
                    borderColor: '#e74c3c',
                    pointBackgroundColor: '#e74c3c',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#e74c3c'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: '#ecf0f1'
                        },
                        ticks: {
                            color: '#bdc3c7',
                            backdropColor: 'transparent'
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ecf0f1'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Risk Factor Analysis',
                        color: '#ecf0f1'
                    }
                }
            }
        });
    }

    updateCharts() {
        // Update decision chart
        if (this.charts.decision && this.currentStats.decision_distribution) {
            const dist = this.currentStats.decision_distribution;
            this.charts.decision.data.datasets[0].data = [dist.allowed || 0, dist.denied || 0, dist.lockout || 0];
            this.charts.decision.update();
        }
        
        // Update time distribution chart
        if (this.charts.time && this.currentStats.time_distribution) {
            this.charts.time.data.datasets[0].data = this.currentStats.time_distribution;
            this.charts.time.update();
        }

        // Embedded dashboard heatmap disabled
    }
    
    updateEventsChart(events) {
        if (!this.charts.scores) return;
        
        // Get last 20 events for trend
        const recentEvents = events.slice(0, 20).reverse();
        
        // Update labels (times)
        const labels = recentEvents.map(event => {
            const date = new Date(event.time);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        });
        
        // Update datasets
        this.charts.scores.data.labels = labels;
        this.charts.scores.data.datasets[0].data = recentEvents.map(e => e.face_score || 0);
        this.charts.scores.data.datasets[1].data = recentEvents.map(e => e.voice_score || 0);
        this.charts.scores.data.datasets[2].data = recentEvents.map(e => e.final_score || 0);
        
        this.charts.scores.update();
    }
    
    toggleChart(chartType) {
        const chartElement = document.getElementById(`${chartType}Chart`);
        if (chartElement) {
            const isVisible = chartElement.style.display !== 'none';
            chartElement.style.display = isVisible ? 'none' : 'block';
            
            // Update chart if it was hidden
            if (!isVisible && this.charts[chartType]) {
                this.charts[chartType].resize();
                this.charts[chartType].update();
            }
        }
    }
    
    applyTimeFilter(timeRange) {
        // Implement time-based filtering
        console.log(`Applying time filter: ${timeRange}`);
        // This would typically make an API call with the filter
    }
    
    applyDecisionFilter(decision) {
        // Implement decision-based filtering
        console.log(`Applying decision filter: ${decision}`);
        // This would typically make an API call with the filter
    }
    
    startAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            this.refreshAll();
        }, 15000); // Refresh every 15 seconds
        
        console.log('🔄 Auto-refresh enabled');
    }
    
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('⏸️ Auto-refresh disabled');
        }
    }
    
    async refreshAll() {
        try {
            await Promise.all([
                this.loadSystemStats(),
                this.loadRecentEvents()
            ]);
            
            // Update last refresh time
            this.updateElement('lastRefresh', new Date().toLocaleTimeString());
            
        } catch (error) {
            console.error('Error refreshing data:', error);
        }
    }
    
    async exportData(format) {
        try {
            const response = await fetch('/api/export');
            const data = await response.json();
            
            if (data.url) {
                // Create download link
                const a = document.createElement('a');
                a.href = data.url;
                a.download = `smart_lock_export_${new Date().toISOString().slice(0,10)}.${format}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                
                this.showToast(`Exported data as ${format.toUpperCase()}`, 'success');
            }
            
        } catch (error) {
            console.error('Error exporting data:', error);
            this.showToast('Export failed', 'error');
        }
    }
    
    showLockoutAlert(reason) {
        // Create lockout alert
        const alert = document.createElement('div');
        alert.className = 'lockout-alert';
        alert.innerHTML = `
            <div class="alert-content">
                <h3>🚨 SYSTEM LOCKOUT</h3>
                <p>${reason}</p>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;
        
        document.body.appendChild(alert);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 10000);
        
        // Play alert sound
        this.playAlertSound();
    }
    
    showSystemAlert(message, level = 'warning') {
        const toast = this.createToast(message, level);
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }
    
    showToast(message, type = 'info') {
        const toast = this.createToast(message, type);
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 3000);
    }
    
    createToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${this.getToastIcon(type)}</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        return toast;
    }
    
    getToastIcon(type) {
        switch(type) {
            case 'success': return '✓';
            case 'error': return '✗';
            case 'warning': return '⚠️';
            default: return 'ℹ️';
        }
    }
    
    getDecisionIcon(decision) {
        switch(decision) {
            case 'ALLOW': return '✅';
            case 'DENY': return '❌';
            case 'LOCKOUT': return '🔒';
            default: return '❓';
        }
    }
    
    updateElement(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
    
    playAlertSound() {
        // Create and play alert sound
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);
            
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
            oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.1);
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.2);
            
            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
            
            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
            
        } catch (error) {
            console.warn('Could not play alert sound:', error);
        }
    }
    
    showNotification(title, body) {
        if ('Notification' in window && Notification.permission === 'granted') {
            new Notification(title, {
                body: body,
                icon: '/static/favicon.ico'
            });
        }
    }
    
    // Clean up resources
    destroy() {
        this.stopAutoRefresh();
        
        if (this.ws) {
            this.ws.close();
        }
        
        // Destroy charts
        Object.values(this.charts).forEach(chart => {
            if (chart && chart.destroy) {
                chart.destroy();
            }
        });
        
        console.log('📊 Dashboard destroyed');
    }
}

// ============================================
// Password Change Feature Functions
// ============================================

let passwordChangeState = {
    email: null,
    otpVerified: false,
    currentStep: 1
};

function openPasswordChangeModal() {
    console.log('🔐 Opening password change modal');
    
    // Reset state
    passwordChangeState = {
        email: 'auto', // Auto-retrieved from server config
        otpVerified: false,
        currentStep: 1
    };
    
    // Show modal
    const modal = document.getElementById('passwordChangeModal');
    modal.style.display = 'flex';
    
    // Show step 2 (skip email input, go directly to OTP)
    showPasswordStep(2);
    
    // Clear all inputs
    document.getElementById('otpInput').value = '';
    document.getElementById('newPinInput').value = '';
    document.getElementById('confirmPinInput').value = '';
    
    // Clear all messages
    // step1 (email) was removed; clear OTP message instead
    document.getElementById('step2-message').innerHTML = '';
    document.getElementById('step3-message').innerHTML = '';
    document.getElementById('success-message').style.display = 'none';
}

function closePasswordChangeModal() {
    console.log('🔐 Closing password change modal');
    const modal = document.getElementById('passwordChangeModal');
    modal.style.display = 'none';
    
    // Reset state
    passwordChangeState = {
        email: null,
        otpVerified: false,
        currentStep: 1
    };
}

function showPasswordStep(stepNumber) {
    console.log(`📍 Showing password change step ${stepNumber}`);
    
    // Hide all steps
    document.getElementById('step2-otp').style.display = 'none';
    document.getElementById('step3-pinchange').style.display = 'none';
    document.getElementById('success-message').style.display = 'none';
    
    // Show requested step
    switch(stepNumber) {
        case 1:
            document.getElementById('step2-otp').style.display = 'block';
            document.getElementById('otpInput').focus();
            break;
        case 2:
            document.getElementById('step3-pinchange').style.display = 'block';
            document.getElementById('newPinInput').focus();
            break;
        case 'success':
            document.getElementById('success-message').style.display = 'block';
            break;
    }
    
    passwordChangeState.currentStep = stepNumber;
}

function clearMessage(messageElementId) {
    const element = document.getElementById(messageElementId);
    element.innerHTML = '';
    element.classList.remove('message-success', 'message-error', 'message-info');
}

function showMessage(messageElementId, message, type = 'info') {
    const element = document.getElementById(messageElementId);
    element.innerHTML = message;
    element.classList.remove('message-success', 'message-error', 'message-info');
    element.classList.add(`message-${type}`);
}

async function requestPasswordOTP() {
    console.log('📧 Requesting OTP (auto email from system config)');
    // OTP is sent automatically to configured email
    
    try {
        showMessage('step2-message', '<i class="fas fa-spinner fa-spin"></i> Sending OTP to registered email...', 'info');

        const response = await fetch('/api/password/request-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (response.ok) {
            passwordChangeState.email = data.email || 'registered';
            showMessage('step2-message', '<i class="fas fa-check-circle"></i> OTP sent to registered email!', 'success');
            document.getElementById('otpInput').focus();
        } else {
            showMessage('step2-message', `<i class="fas fa-times-circle"></i> ${data.message || 'Failed to send OTP'}`, 'error');
        }
    } catch (error) {
        console.error('Error requesting OTP:', error);
        showMessage('step2-message', '<i class="fas fa-times-circle"></i> Network error. Please try again.', 'error');
    }
}

async function verifyPasswordOTP() {
    console.log('🔑 Verifying OTP');
    clearMessage('step2-message');
    
    const otp = document.getElementById('otpInput').value.trim();
    
    // Validate OTP
    if (!otp) {
        showMessage('step2-message', '<i class="fas fa-exclamation-circle"></i> Please enter the OTP', 'error');
        return;
    }
    
    if (!/^\d{6}$/.test(otp)) {
        showMessage('step2-message', '<i class="fas fa-exclamation-circle"></i> OTP must be 6 digits', 'error');
        return;
    }
    
    try {
        showMessage('step2-message', '<i class="fas fa-spinner fa-spin"></i> Verifying OTP...', 'info');
        
        const response = await fetch('/api/password/verify-otp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                otp: otp
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            passwordChangeState.otpVerified = true;
            showMessage('step2-message', '<i class="fas fa-check-circle"></i> OTP verified successfully!', 'success');
            
            // Move to next step after 1.5 seconds
            setTimeout(() => {
                showPasswordStep(2);
            }, 1500);
        } else {
            showMessage('step2-message', `<i class="fas fa-times-circle"></i> ${data.error || 'Invalid OTP'}`, 'error');
        }
    } catch (error) {
        console.error('Error verifying OTP:', error);
        showMessage('step2-message', '<i class="fas fa-times-circle"></i> Network error. Please try again.', 'error');
    }
}

async function changePassword() {
    console.log('🔐 Changing PIN');
    clearMessage('step3-message');
    
    const newPin = document.getElementById('newPinInput').value.trim();
    const confirmPin = document.getElementById('confirmPinInput').value.trim();
    
    // Validate PIN format
    if (!newPin) {
        showMessage('step3-message', '<i class="fas fa-exclamation-circle"></i> Please enter a new PIN', 'error');
        return;
    }
    
    if (!/^\d{4,8}$/.test(newPin)) {
        showMessage('step3-message', '<i class="fas fa-exclamation-circle"></i> PIN must be 4-8 digits', 'error');
        return;
    }
    
    if (!confirmPin) {
        showMessage('step3-message', '<i class="fas fa-exclamation-circle"></i> Please confirm your PIN', 'error');
        return;
    }
    
    if (newPin !== confirmPin) {
        showMessage('step3-message', '<i class="fas fa-exclamation-circle"></i> PINs do not match', 'error');
        return;
    }
    
    try {
        showMessage('step3-message', '<i class="fas fa-spinner fa-spin"></i> Updating PIN...', 'info');
        
        const response = await fetch('/api/password/change', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                new_pin: newPin,
                confirm_pin: confirmPin
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            passwordChangeState.otpVerified = false;
            showPasswordStep('success');
            
            // Auto-close after 4 seconds
            setTimeout(() => {
                closePasswordChangeModal();
                window.dashboard?.showToast('PIN changed successfully! Use your new PIN on next login.', 'success');
            }, 4000);
        } else {
            showMessage('step3-message', `<i class="fas fa-times-circle"></i> ${data.error || 'Failed to change PIN'}`, 'error');
        }
    } catch (error) {
        console.error('Error changing PIN:', error);
        showMessage('step3-message', '<i class="fas fa-times-circle"></i> Network error. Please try again.', 'error');
    }
}

// Close modal when clicking outside of it
window.addEventListener('click', (event) => {
    const modal = document.getElementById('passwordChangeModal');
    if (event.target === modal) {
        closePasswordChangeModal();
    }
});

// Allow Enter key to submit in form inputs
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('emailInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') requestPasswordOTP();
    });
    
    document.getElementById('otpInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') verifyPasswordOTP();
    });
    
    document.getElementById('confirmPinInput')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') changePassword();
    });
});

// --- Lockout Reset OTP Flow ---
async function requestLockoutOTP() {
    document.getElementById('lockout-reset-message').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending OTP to registered email...';
    document.getElementById('lockoutResetStep1').style.display = 'block';
    document.getElementById('lockoutResetStep2').style.display = 'none';
    document.getElementById('lockoutResetStep3').style.display = 'none';
    try {
        const response = await fetch('/api/lockout/request-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await response.json();
        if (response.ok) {
            document.getElementById('lockout-reset-message').innerHTML = '<i class="fas fa-check-circle"></i> OTP sent to registered email!';
            document.getElementById('lockoutResetStep1').style.display = 'none';
            document.getElementById('lockoutResetStep2').style.display = 'block';
            document.getElementById('lockoutOtpInput').focus();
        } else {
            document.getElementById('lockout-reset-message').innerHTML = `<i class="fas fa-times-circle"></i> ${data.message || 'Failed to send OTP'}`;
        }
    } catch (error) {
        document.getElementById('lockout-reset-message').innerHTML = '<i class="fas fa-times-circle"></i> Network error. Please try again.';
    }
}

async function verifyLockoutOTP() {
    const otp = document.getElementById('lockoutOtpInput').value.trim();
    if (!otp) {
        document.getElementById('lockout-otp-message').innerHTML = '<i class="fas fa-exclamation-circle"></i> Please enter the OTP';
        return;
    }
    if (!/^\d{6}$/.test(otp)) {
        document.getElementById('lockout-otp-message').innerHTML = '<i class="fas fa-exclamation-circle"></i> OTP must be 6 digits';
        return;
    }
    document.getElementById('lockout-otp-message').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying OTP...';
    try {
        const response = await fetch('/api/lockout/verify-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ otp })
        });
        const data = await response.json();
        if (response.ok) {
            document.getElementById('lockout-otp-message').innerHTML = '<i class="fas fa-check-circle"></i> OTP verified!';
            document.getElementById('lockoutResetStep2').style.display = 'none';
            document.getElementById('lockoutResetStep3').style.display = 'block';
        } else {
            document.getElementById('lockout-otp-message').innerHTML = `<i class="fas fa-times-circle"></i> ${data.message || 'Invalid OTP'}`;
        }
    } catch (error) {
        document.getElementById('lockout-otp-message').innerHTML = '<i class="fas fa-times-circle"></i> Network error. Please try again.';
    }
}

async function resetLockoutWithOTP() {
    document.getElementById('lockout-final-message').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Unlocking system...';
    try {
        const response = await fetch('/api/lockout/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await response.json();
        if (response.ok) {
            document.getElementById('lockout-final-message').innerHTML = '<i class="fas fa-unlock"></i> System unlocked! Refreshing status...';
            setTimeout(() => { window.location.reload(); }, 2000);
        } else {
            document.getElementById('lockout-final-message').innerHTML = `<i class="fas fa-times-circle"></i> ${data.message || 'Failed to unlock'}`;
        }
    } catch (error) {
        document.getElementById('lockout-final-message').innerHTML = '<i class="fas fa-times-circle"></i> Network error. Please try again.';
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new SmartLockDashboard();
    window.smartLockDashboard = window.dashboard;
    window.refreshDashboard = () => window.dashboard.refreshAll();
    window.exportData = (format) => window.dashboard.exportData(format);
    window.openPasswordChangeModal = openPasswordChangeModal;
    window.closePasswordChangeModal = closePasswordChangeModal;
    window.requestPasswordOTP = requestPasswordOTP;
    window.verifyPasswordOTP = verifyPasswordOTP;
    window.changePassword = changePassword;
    // Lockout reset OTP functions
    window.requestLockoutOTP = requestLockoutOTP;
    window.verifyLockoutOTP = verifyLockoutOTP;
    window.resetLockoutWithOTP = resetLockoutWithOTP;
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            window.dashboard.stopAutoRefresh();
        } else {
            window.dashboard.startAutoRefresh();
        }
    });
});

// Service Worker for push notifications
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js')
        .then(registration => {
            console.log('Service Worker registered with scope:', registration.scope);
        })
        .catch(error => {
            console.error('Service Worker registration failed:', error);
        });
}
